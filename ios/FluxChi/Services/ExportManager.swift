import Foundation
import UIKit
import CryptoKit

enum ExportManager {

    // MARK: - Log Export (委托给 FluxLogger)

    /// 导出日志为 JSON 文件 URL
    @MainActor
    static func shareLogsURL(options: FluxLogExportOptions = .all) throws -> URL {
        try FluxLogger.shared.exportToFile(format: .json, options: options)
    }

    /// 导出日志为文本文件 URL
    @MainActor
    static func shareLogsTextURL(options: FluxLogExportOptions = .all) throws -> URL {
        try FluxLogger.shared.exportToFile(format: .text, options: options)
    }

    // MARK: - Session Export Structures

    struct ExportPackage: Codable {
        let fluxchiVersion: String
        let schemaVersion: Int
        let exportedAt: String
        let device: DeviceInfo
        let session: SessionExport
        let signalMeta: SignalMeta
        let segments: [SegmentExport]
        let snapshots: [SnapshotExport]
        let channelStats: [ChannelStat]
        let feedback: FeedbackExport?
        let summary: SummaryExport?
        let processing: ProcessingInfo
        /// 与网关落盘一致：`fluxchi_yyyyMMdd_HHmmss_{uuid}.json`（导出时刻墙钟 + 全小写 UUID）
        let suggestedArchiveFileName: String?
    }

    struct DeviceInfo: Codable {
        let model: String
        let systemVersion: String
        let source: String
    }

    struct SessionExport: Codable {
        let id: String
        let startedAt: String
        let endedAt: String?
        let title: String
        let source: String
        let durationSec: Double
    }

    struct SignalMeta: Codable {
        let totalChannels: Int
        let activeChannels: Int
        let sampleIntervalMs: Int
        let snapshotCount: Int
        let note: String
    }

    struct SegmentExport: Codable {
        let id: String
        let label: String
        let startedAt: String
        let endedAt: String?
        let durationSec: Double
        let snapshotCount: Int
    }

    struct SnapshotExport: Codable {
        let t: Double
        let segIdx: Int
        let stamina: Double
        let state: String
        let con: Double
        let ten: Double
        let fat: Double
        let act: String
        let conf: Double
        let rms: [Double]
    }

    struct ChannelStat: Codable {
        let channel: Int
        let isActive: Bool
        let meanRMS: Double
        let maxRMS: Double
        let minRMS: Double
        let stdRMS: Double
    }

    struct FeedbackExport: Codable {
        let feeling: String
        let accuracyRating: Int
        let notes: String
    }

    struct SummaryExport: Codable {
        let text: String
        let avgStamina: Double
        let minStamina: Double
        let maxStamina: Double
        let totalDurationSec: Double
        let workDurationSec: Double
        let restDurationSec: Double
    }

    struct ProcessingInfo: Codable {
        let staminaWeights: [String: Double]
        let staminaThresholds: [String: Double]
        let emgDecoding: String
        let featureWindow: String
    }

    struct PlatformSessionUploadBundle {
        let request: PlatformCreateSessionRequest
        let blobData: Data
        let idempotencyKey: String
    }

    enum PlatformUploadPreparationError: LocalizedError {
        case sessionNotFinished
        case missingPlatformDeviceID

        var errorDescription: String? {
            switch self {
            case .sessionNotFinished:
                return "当前 session 尚未结束，无法上传"
            case .missingPlatformDeviceID:
                return "平台设备身份未初始化，请先确保平台同步成功"
            }
        }
    }

    // MARK: - Session Export

    private static func suggestedArchiveFileName(for session: Session, wallClock: Date) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyyMMdd_HHmmss"
        return "fluxchi_\(f.string(from: wallClock))_\(session.id.uuidString.lowercased()).json"
    }

    static func export(session: Session, exportWallClock: Date = Date()) throws -> Data {
        let iso = ISO8601DateFormatter()
        let sessionStart = session.startedAt
        let archiveName = suggestedArchiveFileName(for: session, wallClock: exportWallClock)

        let sortedSegments = session.segments.sorted { $0.startedAt < $1.startedAt }

        var allSnaps: [(snap: FluxSnapshot, segIdx: Int)] = []
        for (segIdx, seg) in sortedSegments.enumerated() {
            for snap in seg.snapshots {
                allSnaps.append((snap, segIdx))
            }
        }
        allSnaps.sort { $0.snap.timestamp < $1.snap.timestamp }

        let activeCount = detectActiveChannels(allSnaps.map(\.snap))

        let snapshotExports: [SnapshotExport] = allSnaps.map { item in
            let rmsSlice = Array(item.snap.rms.prefix(activeCount))
            return SnapshotExport(
                t: item.snap.timestamp.timeIntervalSince(sessionStart),
                segIdx: item.segIdx,
                stamina: round(item.snap.stamina * 10) / 10,
                state: item.snap.stateRaw,
                con: round(item.snap.consistency * 1000) / 1000,
                ten: round(item.snap.tension * 1000) / 1000,
                fat: round(item.snap.fatigue * 1000) / 1000,
                act: item.snap.activity,
                conf: round(item.snap.confidence * 100) / 100,
                rms: rmsSlice.map { round($0 * 100) / 100 }
            )
        }

        let channelStats = computeChannelStats(allSnaps.map(\.snap), totalChannels: 8)

        let package = ExportPackage(
            fluxchiVersion: Flux.App.version,
            schemaVersion: Flux.App.schemaVersion,
            exportedAt: iso.string(from: exportWallClock),
            device: DeviceInfo(
                model: UIDevice.current.model,
                systemVersion: UIDevice.current.systemVersion,
                source: session.sourceRaw
            ),
            session: SessionExport(
                id: session.id.uuidString,
                startedAt: iso.string(from: session.startedAt),
                endedAt: session.endedAt.map { iso.string(from: $0) },
                title: session.title,
                source: session.sourceRaw,
                durationSec: round(session.duration * 10) / 10
            ),
            signalMeta: SignalMeta(
                totalChannels: 8,
                activeChannels: activeCount,
                sampleIntervalMs: Flux.App.snapshotIntervalMs,
                snapshotCount: allSnaps.count,
                note: activeCount < 8
                    ? "BLE mode: 20-byte frame carries 6 channels. Channels 7-8 require USB serial (29-byte frame)."
                    : "USB serial mode: all 8 channels active."
            ),
            segments: sortedSegments.enumerated().map { idx, seg in
                SegmentExport(
                    id: seg.id.uuidString,
                    label: seg.labelRaw,
                    startedAt: iso.string(from: seg.startedAt),
                    endedAt: seg.endedAt.map { iso.string(from: $0) },
                    durationSec: round(seg.duration * 10) / 10,
                    snapshotCount: seg.snapshots.count
                )
            },
            snapshots: snapshotExports,
            channelStats: channelStats,
            feedback: session.feedback.map {
                FeedbackExport(
                    feeling: $0.feelingRaw,
                    accuracyRating: $0.accuracyRating,
                    notes: $0.notes
                )
            },
            summary: session.summaryText.map {
                SummaryExport(
                    text: $0,
                    avgStamina: session.avgStamina ?? 0,
                    minStamina: session.minStamina ?? 0,
                    maxStamina: session.maxStamina ?? 0,
                    totalDurationSec: session.duration,
                    workDurationSec: session.workDurationSec ?? 0,
                    restDurationSec: session.restDurationSec ?? 0
                )
            },
            processing: ProcessingInfo(
                staminaWeights: [
                    "consistency": 0.40,
                    "tension": 0.25,
                    "fatigue": 0.35
                ],
                staminaThresholds: [
                    "focused": 60,
                    "fading": 30,
                    "depleted": 0
                ],
                emgDecoding: "24-bit signed -> uV = (value / 8388607) * 4.5 / 1200 * 1e6",
                featureWindow: "250 samples @ 1kHz = 250ms sliding window"
            ),
            suggestedArchiveFileName: archiveName
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(package)
    }

    static func shareURL(for session: Session) throws -> URL {
        let wall = Date()
        let data = try export(session: session, exportWallClock: wall)
        let filename = suggestedArchiveFileName(for: session, wallClock: wall)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        try data.write(to: url)
        return url
    }

    static func makePlatformSessionUploadBundle(
        for session: Session,
        platformDeviceID: String
    ) throws -> PlatformSessionUploadBundle {
        guard let endedAt = session.endedAt else {
            throw PlatformUploadPreparationError.sessionNotFinished
        }
        guard !platformDeviceID.isEmpty else {
            throw PlatformUploadPreparationError.missingPlatformDeviceID
        }

        let blobData = try export(session: session)
        let allSnapshots = session.segments.reduce(0) { $0 + $1.snapshots.count }
        let sessionID = platformSessionID(for: session)

        return PlatformSessionUploadBundle(
            request: PlatformCreateSessionRequest(
                sessionID: sessionID,
                deviceID: platformDeviceID,
                source: platformSource(for: session.source),
                title: session.title.isEmpty ? nil : session.title,
                startedAt: session.startedAt,
                endedAt: endedAt,
                durationSec: max(0, Int(session.duration.rounded())),
                snapshotCount: allSnapshots,
                schemaVersion: session.schemaVersion,
                contentType: "application/json",
                sizeBytes: blobData.count,
                sha256: sha256Hex(for: blobData)
            ),
            blobData: blobData,
            idempotencyKey: "ios:session-upload:\(session.id.uuidString.lowercased())"
        )
    }

    // MARK: - Helpers

    private static func platformSessionID(for session: Session) -> String {
        "ses_\(session.id.uuidString.lowercased().replacingOccurrences(of: "-", with: ""))"
    }

    private static func platformSource(for source: SessionSource) -> String {
        switch source {
        case .ble:
            return "ios_ble"
        case .wifi:
            return "ios_wifi"
        }
    }

    private static func sha256Hex(for data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func detectActiveChannels(_ snapshots: [FluxSnapshot]) -> Int {
        guard !snapshots.isEmpty else { return 6 }
        let sampleCount = min(snapshots.count, 100)
        let sampled = snapshots.suffix(sampleCount)
        let ch7HasData = sampled.contains { abs($0.rms6) > 0.01 }
        let ch8HasData = sampled.contains { abs($0.rms7) > 0.01 }
        return (ch7HasData || ch8HasData) ? 8 : 6
    }

    private static func computeChannelStats(_ snapshots: [FluxSnapshot], totalChannels: Int) -> [ChannelStat] {
        guard !snapshots.isEmpty else {
            return (0..<totalChannels).map { ChannelStat(channel: $0 + 1, isActive: false, meanRMS: 0, maxRMS: 0, minRMS: 0, stdRMS: 0) }
        }

        return (0..<totalChannels).map { ch in
            let values = snapshots.map { snap -> Double in
                switch ch {
                case 0: return snap.rms0
                case 1: return snap.rms1
                case 2: return snap.rms2
                case 3: return snap.rms3
                case 4: return snap.rms4
                case 5: return snap.rms5
                case 6: return snap.rms6
                case 7: return snap.rms7
                default: return 0
                }
            }

            let isActive = values.contains { abs($0) > 0.01 }
            guard isActive else {
                return ChannelStat(channel: ch + 1, isActive: false, meanRMS: 0, maxRMS: 0, minRMS: 0, stdRMS: 0)
            }

            let n = Double(values.count)
            let mean = values.reduce(0, +) / n
            let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / n
            let std = sqrt(variance)

            return ChannelStat(
                channel: ch + 1,
                isActive: true,
                meanRMS: round(mean * 100) / 100,
                maxRMS: round((values.max() ?? 0) * 100) / 100,
                minRMS: round((values.min() ?? 0) * 100) / 100,
                stdRMS: round(std * 100) / 100
            )
        }
    }

    // MARK: - Gateway sync（与 Python 网关归档对齐）

    /// 大 JSON ingest：不用 `URLSession.shared` 的默认超时，避免弱网下误杀。
    private static let ingestURLSession: URLSession = {
        let c = URLSessionConfiguration.default
        c.timeoutIntervalForRequest = 60
        c.timeoutIntervalForResource = 300
        c.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: c)
    }()

    enum GatewaySyncError: LocalizedError {
        case invalidURL
        case httpStatus(Int, String)
        case badEnvelope

        var errorDescription: String? {
            switch self {
            case .invalidURL: return "网关地址无效"
            case .httpStatus(let c, let t): return "网关 HTTP \(c): \(t)"
            case .badEnvelope: return "网关返回格式异常"
            }
        }
    }

    private struct IngestEnvelope: Decodable {
        let ok: Bool?
        let data: IngestData?
    }

    private struct IngestData: Decodable {
        let sessionId: String?
        let insight: String?

        enum CodingKeys: String, CodingKey {
            case sessionId = "session_id"
            case insight
        }
    }

    /// 将导出 JSON POST 到运行 `web/app.py` 的机器；后端统一 8 路 RMS 并生成洞察。
    static func uploadSessionToGateway(session: Session) async throws -> String {
        let data = try export(session: session)
        let host = UserDefaults.standard.string(forKey: "flux_host") ?? "127.0.0.1"
        var port = UserDefaults.standard.integer(forKey: "flux_port")
        if port == 0 { port = 8000 }
        return try await uploadExportData(data, host: host, port: port)
    }

    static func uploadExportData(_ data: Data, host: String, port: Int) async throws -> String {
        let urlStr = "http://\(host):\(port)/api/v1/sessions/ingest"
        guard let url = URL(string: urlStr) else { throw GatewaySyncError.invalidURL }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json; charset=utf-8", forHTTPHeaderField: "Content-Type")
        req.httpBody = data
        let (respData, resp) = try await ingestURLSession.data(for: req)
        guard let http = resp as? HTTPURLResponse else { throw GatewaySyncError.badEnvelope }
        let preview = String(data: respData, encoding: .utf8).map { String($0.prefix(200)) } ?? ""
        guard (200...299).contains(http.statusCode) else {
            throw GatewaySyncError.httpStatus(http.statusCode, preview)
        }
        let dec = JSONDecoder()
        let env = try? dec.decode(IngestEnvelope.self, from: respData)
        guard env?.ok == true, let sid = env?.data?.sessionId, !sid.isEmpty else {
            throw GatewaySyncError.badEnvelope
        }
        return sid
    }
}
