import Foundation
import UIKit

enum ExportManager {

    // MARK: - Log Export

    /// 日志导出选项
    struct LogExportOptions {
        let minimumLevel: FluxLogLevel?
        let categories: Set<FluxLogCategory>?
        let limit: Int?
        let keyword: String?

        static let `default` = LogExportOptions(
            minimumLevel: nil,
            categories: nil,
            limit: nil,
            keyword: nil
        )

        static let errorsOnly = LogExportOptions(
            minimumLevel: .error,
            categories: nil,
            limit: nil,
            keyword: nil
        )

        static let bleOnly = LogExportOptions(
            minimumLevel: nil,
            categories: [.ble],
            limit: nil,
            keyword: nil
        )
    }

    /// 导出日志为 JSON 数据
    static func exportLogs(options: LogExportOptions = .default) throws -> Data {
        let logger = FluxLogger.shared

        var filtered = logger.entries

        // 按级别过滤
        if let minLevel = options.minimumLevel {
            filtered = filtered.filter { $0.level >= minLevel }
        }

        // 按分类过滤
        if let categories = options.categories {
            filtered = filtered.filter { categories.contains($0.category) }
        }

        // 按关键词搜索
        if let keyword = options.keyword, !keyword.isEmpty {
            filtered = filtered.filter {
                $0.message.localizedCaseInsensitiveContains(keyword) ||
                $0.errorDescription?.localizedCaseInsensitiveContains(keyword) == true
            }
        }

        // 限制数量
        if let limit = options.limit {
            filtered = Array(filtered.suffix(limit))
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        return try encoder.encode(filtered)
    }

    /// 导出日志为纯文本
    static func exportLogsAsText(options: LogExportOptions = .default) throws -> String {
        let data = try exportLogs(options: options)
        let entries = try JSONDecoder().decode([FluxLogEntry].self, from: data)

        return entries.map { entry in
            entry.formatDetailed()
        }.joined(separator: "\n---\n")
    }

    /// 分享日志 JSON 文件 URL
    static func shareLogsURL(options: LogExportOptions = .default) throws -> URL {
        let data = try exportLogs(options: options)

        let dateStr = {
            let f = DateFormatter()
            f.dateFormat = "yyyyMMdd_HHmmss"
            return f.string(from: Date())
        }()

        let filename = "fluxchi_logs_\(dateStr).json"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)

        try data.write(to: url)
        return url
    }

    /// 分享日志文本文件 URL
    static func shareLogsTextURL(options: LogExportOptions = .default) throws -> URL {
        let text = try exportLogsAsText(options: options)

        let dateStr = {
            let f = DateFormatter()
            f.dateFormat = "yyyyMMdd_HHmmss"
            return f.string(from: Date())
        }()

        let filename = "fluxchi_logs_\(dateStr).txt"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)

        try text.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    // MARK: - Export Structures

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

    // MARK: - Export

    static func export(session: Session) throws -> Data {
        let iso = ISO8601DateFormatter()
        let sessionStart = session.startedAt

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
            exportedAt: iso.string(from: Date()),
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
                emgDecoding: "24-bit signed → μV = (value / 8388607) × 4.5 / 1200 × 1e6",
                featureWindow: "250 samples @ 1kHz = 250ms sliding window"
            )
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(package)
    }

    static func shareURL(for session: Session) throws -> URL {
        let data = try export(session: session)
        let dateStr = {
            let f = DateFormatter()
            f.dateFormat = "yyyyMMdd_HHmm"
            return f.string(from: session.startedAt)
        }()
        let filename = "fluxchi_\(dateStr)_\(session.id.uuidString.prefix(6)).json"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        try data.write(to: url)
        return url
    }

    // MARK: - Helpers

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
}
