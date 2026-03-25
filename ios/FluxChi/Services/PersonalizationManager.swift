import Foundation
import UIKit

@MainActor
final class PersonalizationManager: ObservableObject {

    @Published var trainingCount: Int = 0
    @Published var estimatedAccuracy: Double = 0

    /// Learned offset applied to raw stamina predictions.
    @Published private(set) var calibrationOffset: Double = 0
    @Published private(set) var isSyncing = false
    @Published private(set) var syncStatusMessage: String?
    @Published private(set) var lastSyncAt: Date?

    private let alpha: Double = 0.3
    private let retainedFeedbackLimit = 500
    private var feedbackPairs: [FeedbackPair] = []
    private var deviceCalibrations: [String: DeviceCalibration] = [:]
    private var profileID: String = ""
    private var deviceID: String = ""
    private var profileUpdatedAt: Date?

    /// 平台 API 入口，用于画像同步和 feedback-events 写入。
    weak var fluxService: FluxService?

    private static let feedbackDataFileName = "feedback_pairs.json"
    private static let deviceCalibrationDefaultsKey = "flux_ml_device_calibrations"
    private static let profileIDDefaultsKey = "flux_ml_profile_id"
    private static let deviceIDDefaultsKey = "flux_ml_device_id"
    private static let profileUpdatedAtDefaultsKey = "flux_ml_profile_updated_at"
    private static let lastSyncAtDefaultsKey = "flux_ml_last_sync_at"

    private var dataDir: URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("FluxModels", isDirectory: true)
    }

    private var feedbackDataURL: URL {
        dataDir.appendingPathComponent(Self.feedbackDataFileName)
    }

    private enum SyncDirection {
        case push
        case pull
    }

    init() {
        try? FileManager.default.createDirectory(at: dataDir, withIntermediateDirectories: true)
        loadState()
        loadFeedbackPairs()
        loadDeviceCalibrations()
        ensureStableIDs()
        updateCurrentDeviceCalibration(updatedAt: resolvedProfileUpdatedAt())
        trainingCount = max(trainingCount, feedbackPairs.count)
        if estimatedAccuracy <= 0, !feedbackPairs.isEmpty {
            recalculateEstimatedAccuracy()
        }
        saveDeviceCalibrations()
        saveState()
    }

    // MARK: - Personalized Prediction

    func personalizedStamina(_ rawValue: Double) -> Double {
        max(0, min(100, rawValue + calibrationOffset))
    }

    // MARK: - Feedback Collection

    /// 接收 session 反馈，更新标定偏移。同一 session 只学习一次（持久化去重）。
    func addTrainingData(session: Session, feedback: UserFeedback) {
        let sid = session.id.uuidString
        let predicted = session.avgStamina ?? 50
        let actual = feedback.feeling.staminaTarget
        let alreadyLearned = feedbackPairs.contains(where: { $0.sessionID == sid })

        if alreadyLearned {
            FluxLog.ml.info("Session \(sid.prefix(8)) 已学习过，跳过重复训练")
        } else {
            let pair = FeedbackPair(sessionID: sid, predicted: predicted, actual: actual, createdAt: Date())
            feedbackPairs.append(pair)
            feedbackPairs = Array(feedbackPairs.suffix(retainedFeedbackLimit))
            trainingCount += 1

            let error = actual - predicted
            calibrationOffset = calibrationOffset * (1 - alpha) + error * alpha
            profileUpdatedAt = Date()
            updateCurrentDeviceCalibration(updatedAt: resolvedProfileUpdatedAt())

            recalculateEstimatedAccuracy()

            saveFeedbackPairs()
            saveDeviceCalibrations()
            saveState()
        }

        Task { @MainActor [weak self] in
            await self?.postFeedbackEventToPlatform(session: session, feedback: feedback, predicted: predicted, actual: actual)
        }
    }

    // MARK: - Profile Sync

    func pushProfileToServer() async {
        await syncProfile(direction: .push, silentErrors: false)
    }

    func pullProfileFromServer() async {
        await syncProfile(direction: .pull, silentErrors: false)
    }

    func bootstrapFromServerSilently() async {
        await syncProfile(direction: .pull, silentErrors: true)
    }

    // MARK: - Platform Feedback

    /// 将反馈写入平台 feedback-events；若 session 尚未上传，则降级为不带 session_id 的事件。
    private func postFeedbackEventToPlatform(
        session: Session,
        feedback: UserFeedback,
        predicted: Double,
        actual: Double
    ) async {
        guard let service = fluxService else { return }

        do {
            let deviceID = try await service.resolvePlatformDeviceID()
            let eventID = platformFeedbackEventID(for: session)
            let idempotencyKey = platformFeedbackIdempotencyKey(for: session)
            let sessionID = ExportManager.platformSessionID(for: session)
            let predictedValue = Int(max(0, min(100, predicted.rounded())))
            let actualValue = Int(max(0, min(100, actual.rounded())))

            let payload = PlatformCreateFeedbackEventRequest(
                feedbackEventID: eventID,
                deviceID: deviceID,
                sessionID: sessionID,
                predictedStamina: predictedValue,
                actualStamina: actualValue,
                label: platformFeedbackLabel(for: actualValue),
                kss: platformKSS(for: actualValue),
                note: normalizedFeedbackNote(feedback.notes),
                createdAt: feedback.createdAt
            )

            do {
                _ = try await service.createPlatformFeedbackEvent(payload, idempotencyKey: idempotencyKey)
                FluxLog.network.info("feedback-event 已写入平台: \(eventID)")
            } catch {
                guard shouldRetryFeedbackWithoutSession(error) else {
                    throw error
                }

                let fallbackPayload = PlatformCreateFeedbackEventRequest(
                    feedbackEventID: eventID,
                    deviceID: deviceID,
                    sessionID: nil,
                    predictedStamina: predictedValue,
                    actualStamina: actualValue,
                    label: platformFeedbackLabel(for: actualValue),
                    kss: platformKSS(for: actualValue),
                    note: normalizedFeedbackNote(feedback.notes),
                    createdAt: feedback.createdAt
                )
                _ = try await service.createPlatformFeedbackEvent(fallbackPayload, idempotencyKey: idempotencyKey)
                FluxLog.network.info("feedback-event 已写入平台（无 session 关联）: \(eventID)")
            }
        } catch {
            FluxLog.network.warn("feedback-event 写入平台失败: \(error.localizedDescription)")
        }
    }

    // MARK: - Persistence

    private func loadState() {
        let d = UserDefaults.standard
        trainingCount = d.integer(forKey: "flux_ml_count")
        calibrationOffset = d.double(forKey: "flux_ml_offset")
        estimatedAccuracy = d.double(forKey: "flux_ml_accuracy")
        let updatedAt = d.double(forKey: Self.profileUpdatedAtDefaultsKey)
        if updatedAt > 0 {
            profileUpdatedAt = Date(timeIntervalSince1970: updatedAt)
        }
        let syncedAt = d.double(forKey: Self.lastSyncAtDefaultsKey)
        if syncedAt > 0 {
            lastSyncAt = Date(timeIntervalSince1970: syncedAt)
        }
        profileID = d.string(forKey: Self.profileIDDefaultsKey) ?? ""
        deviceID = d.string(forKey: Self.deviceIDDefaultsKey) ?? ""
    }

    private func saveState() {
        let d = UserDefaults.standard
        d.set(trainingCount, forKey: "flux_ml_count")
        d.set(calibrationOffset, forKey: "flux_ml_offset")
        d.set(estimatedAccuracy, forKey: "flux_ml_accuracy")
        d.set(profileID, forKey: Self.profileIDDefaultsKey)
        d.set(deviceID, forKey: Self.deviceIDDefaultsKey)
        d.set(profileUpdatedAt?.timeIntervalSince1970 ?? 0, forKey: Self.profileUpdatedAtDefaultsKey)
        d.set(lastSyncAt?.timeIntervalSince1970 ?? 0, forKey: Self.lastSyncAtDefaultsKey)
    }

    private func loadFeedbackPairs() {
        guard let data = try? Data(contentsOf: feedbackDataURL),
              let decoded = try? Self.makeDecoder().decode([FeedbackPair].self, from: data) else { return }
        feedbackPairs = Array(decoded.suffix(retainedFeedbackLimit))
    }

    private func saveFeedbackPairs() {
        guard let data = try? Self.makeEncoder().encode(feedbackPairs) else { return }
        try? data.write(to: feedbackDataURL)
    }

    private func loadDeviceCalibrations() {
        let defaults = UserDefaults.standard
        guard let data = defaults.data(forKey: Self.deviceCalibrationDefaultsKey),
              let decoded = try? Self.makeDecoder().decode([String: DeviceCalibration].self, from: data) else { return }
        deviceCalibrations = decoded
    }

    private func saveDeviceCalibrations() {
        guard let data = try? Self.makeEncoder().encode(deviceCalibrations) else { return }
        UserDefaults.standard.set(data, forKey: Self.deviceCalibrationDefaultsKey)
    }

    private func ensureStableIDs() {
        if profileID.isEmpty {
            profileID = UUID().uuidString
        }
        if deviceID.isEmpty {
            deviceID = UserDefaults.standard.string(forKey: Self.deviceIDDefaultsKey)
                ?? UIDevice.current.identifierForVendor?.uuidString
                ?? UUID().uuidString
        }
    }

    private func resolvedProfileUpdatedAt() -> Date {
        if let profileUpdatedAt {
            return profileUpdatedAt
        }
        let fallback = feedbackPairs.compactMap(\.createdAt).max() ?? Date()
        profileUpdatedAt = fallback
        return fallback
    }

    private func updateCurrentDeviceCalibration(updatedAt: Date) {
        guard !deviceID.isEmpty else { return }
        deviceCalibrations[deviceID] = DeviceCalibration(
            deviceID: deviceID,
            deviceName: UIDevice.current.name,
            calibrationOffset: calibrationOffset,
            updatedAt: updatedAt
        )
    }

    private func recalculateEstimatedAccuracy() {
        let recent = feedbackPairs.suffix(10)
        guard !recent.isEmpty else {
            estimatedAccuracy = 0
            return
        }
        let avgError = recent.map { abs($0.predicted + calibrationOffset - $0.actual) }
            .reduce(0, +) / Double(recent.count)
        estimatedAccuracy = max(0, min(100, 100 - avgError))
    }

    // MARK: - Platform Sync

    private func syncProfile(direction: SyncDirection, silentErrors: Bool) async {
        guard let service = fluxService else {
            if !silentErrors {
                syncStatusMessage = "服务器未初始化"
            }
            return
        }

        isSyncing = true
        if !silentErrors {
            syncStatusMessage = direction == .push ? "正在上传个性化数据…" : "正在拉取个性化数据…"
        }
        defer { isSyncing = false }

        do {
            let bootstrap = try await service.fetchPlatformBootstrap()
            if let platformDeviceID = service.currentPlatformDeviceID, !platformDeviceID.isEmpty {
                deviceID = platformDeviceID
            }

            switch direction {
            case .pull:
                let applied = applyPlatformBootstrap(bootstrap)
                if !silentErrors {
                    syncStatusMessage = applied ? "已从平台拉取" : "本地版本更新，已保留本地画像"
                }
                FluxLog.network.info("平台 bootstrap 已同步到本地")

            case .push:
                let updatedProfile = try await service.updatePlatformProfile(
                    baseVersion: bootstrap.profileState.version,
                    calibrationOffset: calibrationOffset,
                    estimatedAccuracy: estimatedAccuracy,
                    trainingCount: trainingCount,
                    activeModelReleaseID: bootstrap.profileState.activeModelReleaseID,
                    summary: buildPlatformSummary()
                )

                let currentDeviceID = service.currentPlatformDeviceID ?? deviceID
                guard let currentCalibration = bootstrap.deviceCalibrations.first(where: { $0.deviceID == currentDeviceID }) else {
                    throw FluxServiceError.envelopeFailed(
                        code: "device_not_found",
                        message: "服务器未返回当前设备的校准状态"
                    )
                }

                let updatedCalibration = try await service.updatePlatformDeviceCalibration(
                    deviceID: currentDeviceID,
                    baseVersion: currentCalibration.version,
                    deviceName: UIDevice.current.name,
                    sensorProfile: buildPlatformSensorProfile(),
                    calibrationOffset: calibrationOffset
                )

                _ = applyPlatformProfileState(updatedProfile, allowOlder: true)
                mergePlatformDeviceCalibrations([updatedCalibration], allowOlder: true)
                if !silentErrors {
                    syncStatusMessage = "已上传到平台"
                }
                FluxLog.network.info("个性化画像已上传到平台")
            }

            lastSyncAt = Date()
            saveFeedbackPairs()
            saveDeviceCalibrations()
            saveState()
        } catch {
            if !silentErrors {
                syncStatusMessage = error.localizedDescription
            }
            FluxLog.network.warn("个性化同步失败: \(error.localizedDescription)")
        }
    }

    @discardableResult
    private func applyPlatformBootstrap(_ bootstrap: PlatformBootstrapData) -> Bool {
        mergePlatformDeviceCalibrations(bootstrap.deviceCalibrations)
        return applyPlatformProfileState(bootstrap.profileState)
    }

    @discardableResult
    private func applyPlatformProfileState(_ state: PlatformProfileState, allowOlder: Bool = false) -> Bool {
        let localUpdatedAt = resolvedProfileUpdatedAt()
        if !allowOlder && state.updatedAt < localUpdatedAt {
            return false
        }

        profileID = state.profileID
        calibrationOffset = state.calibrationOffset
        trainingCount = max(trainingCount, state.trainingCount)
        estimatedAccuracy = state.estimatedAccuracy
        profileUpdatedAt = state.updatedAt

        if estimatedAccuracy <= 0, !feedbackPairs.isEmpty {
            recalculateEstimatedAccuracy()
        }

        saveState()
        return true
    }

    private func mergePlatformDeviceCalibrations(
        _ remoteCalibrations: [PlatformDeviceCalibrationState],
        allowOlder: Bool = false
    ) {
        for remote in remoteCalibrations {
            if let existing = deviceCalibrations[remote.deviceID],
               !allowOlder,
               remote.updatedAt < existing.updatedAt {
                continue
            }
            deviceCalibrations[remote.deviceID] = DeviceCalibration(
                deviceID: remote.deviceID,
                deviceName: remote.deviceName,
                calibrationOffset: remote.calibrationOffset,
                updatedAt: remote.updatedAt
            )
        }
    }

    private func buildPlatformSummary() -> PlatformProfileSummary {
        let retained = Array(feedbackPairs.suffix(retainedFeedbackLimit))
        let avgAbsoluteError: Double?
        if retained.isEmpty {
            avgAbsoluteError = nil
        } else {
            avgAbsoluteError = retained
                .map { abs($0.predicted + calibrationOffset - $0.actual) }
                .reduce(0, +) / Double(retained.count)
        }

        return PlatformProfileSummary(
            retainedFeedbackCount: retained.count,
            avgAbsoluteError: avgAbsoluteError,
            lastFeedbackAt: retained.compactMap(\.createdAt).max()
        )
    }

    private func buildPlatformSensorProfile() -> PlatformSensorProfile {
        let calibration = EMGCalibrationStore.load()
        return PlatformSensorProfile(
            channels: EMGCalibrationStore.channelCount,
            sampleRateHz: 1000,
            quality: calibration?.quality,
            relaxMean: calibration?.relaxMean,
            mvcPeak: calibration?.mvcPeak,
            calibratedAt: calibration?.calibratedAt.map { Date(timeIntervalSince1970: $0) }
        )
    }

    private func platformFeedbackEventID(for session: Session) -> String {
        "fbk_\(session.id.uuidString.lowercased().replacingOccurrences(of: "-", with: ""))"
    }

    private func platformFeedbackIdempotencyKey(for session: Session) -> String {
        "ios:feedback-upload:\(session.id.uuidString.lowercased())"
    }

    private func platformFeedbackLabel(for actualStamina: Int) -> String {
        actualStamina < 50 ? "fatigued" : "alert"
    }

    private func platformKSS(for actualStamina: Int) -> Int {
        switch actualStamina {
        case 80...:
            return 2
        case 50..<80:
            return 5
        default:
            return 8
        }
    }

    private func normalizedFeedbackNote(_ note: String) -> String? {
        let trimmed = note.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func shouldRetryFeedbackWithoutSession(_ error: Error) -> Bool {
        guard case let FluxServiceError.envelopeFailed(code, _) = error else {
            return false
        }
        return code == "resource_not_found"
    }

    private static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    private static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
