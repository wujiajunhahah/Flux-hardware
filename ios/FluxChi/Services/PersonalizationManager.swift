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

    /// WiFi 模式下回传反馈给服务端飞轮
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

    private static let syncSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 10
        config.timeoutIntervalForResource = 30
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: config)
    }()

    private enum SyncError: LocalizedError {
        case serviceUnavailable
        case invalidResponse
        case httpStatus(Int, String)
        case badEnvelope
        case server(String)

        var errorDescription: String? {
            switch self {
            case .serviceUnavailable:
                return "服务器未初始化"
            case .invalidResponse:
                return "无效的服务器响应"
            case .httpStatus(let code, let body):
                return body.isEmpty ? "HTTP \(code)" : "HTTP \(code): \(body)"
            case .badEnvelope:
                return "服务器返回格式异常"
            case .server(let message):
                return message.isEmpty ? "服务返回错误" : message
            }
        }
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
        if feedbackPairs.contains(where: { $0.sessionID == sid }) {
            FluxLog.ml.info("Session \(sid.prefix(8)) 已学习过，跳过重复训练")
            return
        }

        let predicted = session.avgStamina ?? 50
        let actual = feedback.feeling.staminaTarget

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

        postFeedbackToFlywheel(predicted: predicted, actual: actual)
    }

    // MARK: - Profile Sync

    func pushProfileToServer() async {
        await syncProfile(direction: .push)
    }

    func pullProfileFromServer() async {
        await syncProfile(direction: .pull)
    }

    // MARK: - Flywheel Integration

    /// 将反馈回传服务端数据飞轮，让全局模型也能从用户反馈中学习
    private func postFeedbackToFlywheel(predicted: Double, actual: Double) {
        guard let service = fluxService else { return }
        let label = actual < 50 ? "fatigued" : "alert"
        let kss: Int
        switch actual {
        case 80...: kss = 2
        case 50..<80: kss = 5
        default: kss = 8
        }

        let url = service.baseURL.appendingPathComponent("api/v1/flywheel/label")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 5
        let body: [String: Any] = [
            "label": label,
            "kss": kss,
            "note": "ios_feedback predicted=\(Int(predicted)) actual=\(Int(actual))"
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        Task.detached {
            _ = try? await URLSession.shared.data(for: request)
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

    private func buildProfile() -> PersonalizationProfile {
        let updatedAt = resolvedProfileUpdatedAt()
        updateCurrentDeviceCalibration(updatedAt: updatedAt)
        let retained = Array(feedbackPairs.suffix(retainedFeedbackLimit))
        let avgAbsoluteError: Double
        if retained.isEmpty {
            avgAbsoluteError = 0
        } else {
            avgAbsoluteError = retained.map { abs($0.predicted + calibrationOffset - $0.actual) }
                .reduce(0, +) / Double(retained.count)
        }
        let lastSessionID = retained.reversed().first(where: { !$0.sessionID.isEmpty })?.sessionID

        return PersonalizationProfile(
            profileID: profileID,
            updatedAt: updatedAt,
            trainingCount: trainingCount,
            estimatedAccuracy: estimatedAccuracy,
            calibrationOffset: calibrationOffset,
            feedbackSummary: PersonalizationFeedbackSummary(
                totalCount: trainingCount,
                retainedCount: retained.count,
                avgAbsoluteError: avgAbsoluteError,
                lastSessionID: lastSessionID
            ),
            recentFeedbackPairs: retained,
            deviceCalibrations: deviceCalibrations
        )
    }

    @discardableResult
    private func applyRemoteProfile(_ profile: PersonalizationProfile, allowOlder: Bool = false) -> Bool {
        let localUpdatedAt = resolvedProfileUpdatedAt()
        if !allowOlder && profile.updatedAt < localUpdatedAt {
            return false
        }

        profileID = profile.profileID
        calibrationOffset = profile.calibrationOffset
        trainingCount = max(profile.trainingCount, profile.recentFeedbackPairs.count)
        estimatedAccuracy = profile.estimatedAccuracy
        feedbackPairs = Array(profile.recentFeedbackPairs.suffix(retainedFeedbackLimit))
        deviceCalibrations = profile.deviceCalibrations
        profileUpdatedAt = profile.updatedAt
        updateCurrentDeviceCalibration(updatedAt: profile.updatedAt)
        if estimatedAccuracy <= 0, !feedbackPairs.isEmpty {
            recalculateEstimatedAccuracy()
        }

        saveFeedbackPairs()
        saveDeviceCalibrations()
        saveState()
        return true
    }

    private enum SyncDirection {
        case push
        case pull
    }

    private func syncProfile(direction: SyncDirection) async {
        guard let service = fluxService else {
            syncStatusMessage = SyncError.serviceUnavailable.localizedDescription
            return
        }

        isSyncing = true
        syncStatusMessage = direction == .push ? "正在上传个性化数据…" : "正在拉取个性化数据…"
        defer { isSyncing = false }

        do {
            switch direction {
            case .push:
                try await pushProfile(baseURL: service.baseURL)
            case .pull:
                try await pullProfile(baseURL: service.baseURL)
            }
            lastSyncAt = Date()
            saveState()
        } catch {
            syncStatusMessage = error.localizedDescription
            FluxLog.network.warn("个性化同步失败: \(error.localizedDescription)")
        }
    }

    private func pushProfile(baseURL: URL) async throws {
        let profile = buildProfile()
        let payload = try Self.makeEncoder().encode(profile)
        let envelope: FluxResponse<PersonalizationProfilePutData> = try await requestEnvelope(
            path: "api/v1/profile",
            method: "PUT",
            baseURL: baseURL,
            body: payload
        )
        guard envelope.ok else {
            throw SyncError.server(envelope.message ?? envelope.error ?? "")
        }
        guard let data = envelope.data else {
            throw SyncError.badEnvelope
        }

        if data.applied {
            if let remote = data.profile {
                _ = applyRemoteProfile(remote, allowOlder: true)
            }
            syncStatusMessage = "已上传到服务器"
            FluxLog.network.info("个性化画像已上传")
            return
        }

        if let remote = data.profile, applyRemoteProfile(remote) {
            syncStatusMessage = "服务器版本较新，已同步到本地"
            FluxLog.network.info("服务器画像较新，已覆盖本地")
        } else {
            syncStatusMessage = "服务器版本较新，本地未覆盖"
            FluxLog.network.info("服务器画像较新，本地保持不变")
        }
    }

    private func pullProfile(baseURL: URL) async throws {
        let envelope: FluxResponse<PersonalizationProfileGetData> = try await requestEnvelope(
            path: "api/v1/profile",
            method: "GET",
            baseURL: baseURL
        )
        guard envelope.ok else {
            throw SyncError.server(envelope.message ?? envelope.error ?? "")
        }
        guard let data = envelope.data else {
            throw SyncError.badEnvelope
        }
        guard data.exists, let profile = data.profile else {
            syncStatusMessage = "服务器上还没有个性化数据"
            FluxLog.network.info("服务器上暂无个性化画像")
            return
        }

        if applyRemoteProfile(profile) {
            syncStatusMessage = "已从服务器拉取"
            FluxLog.network.info("个性化画像已从服务器拉取")
        } else {
            syncStatusMessage = "本地版本更新，未覆盖"
            FluxLog.network.info("远端画像较旧，保留本地版本")
        }
    }

    private func requestEnvelope<T: Decodable>(
        path: String,
        method: String,
        baseURL: URL,
        body: Data? = nil
    ) async throws -> FluxResponse<T> {
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = method
        if body != nil {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = body
        }

        let (data, response) = try await Self.syncSession.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw SyncError.invalidResponse
        }
        let preview = String(data: data, encoding: .utf8).map { String($0.prefix(200)) } ?? ""
        guard (200...299).contains(http.statusCode) else {
            throw SyncError.httpStatus(http.statusCode, preview)
        }
        guard let envelope = try? Self.makeDecoder().decode(FluxResponse<T>.self, from: data) else {
            throw SyncError.badEnvelope
        }
        return envelope
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
