import Foundation
import UIKit

enum ExportManager {

    // MARK: - Export Structures

    struct ExportPackage: Codable {
        let fluxchiVersion: String
        let schemaVersion: Int
        let exportedAt: String
        let device: DeviceInfo
        let session: SessionExport
        let segments: [SegmentExport]
        let snapshots: [SnapshotExport]
        let feedback: FeedbackExport?
        let summary: SummaryExport?
    }

    struct DeviceInfo: Codable {
        let model: String
        let systemVersion: String
    }

    struct SessionExport: Codable {
        let id: String
        let startedAt: String
        let endedAt: String?
        let title: String
        let source: String
        let durationSec: Double
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
        let timestamp: String
        let segmentId: String
        let stamina: Double
        let state: String
        let consistency: Double
        let tension: Double
        let fatigue: Double
        let activity: String
        let confidence: Double
        let rms: [Double]
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

    // MARK: - Export

    static func export(session: Session) throws -> Data {
        let iso = ISO8601DateFormatter()

        let allSnapshots: [SnapshotExport] = session.segments.flatMap { seg in
            seg.snapshots.map { snap in
                SnapshotExport(
                    timestamp: iso.string(from: snap.timestamp),
                    segmentId: seg.id.uuidString,
                    stamina: snap.stamina,
                    state: snap.stateRaw,
                    consistency: snap.consistency,
                    tension: snap.tension,
                    fatigue: snap.fatigue,
                    activity: snap.activity,
                    confidence: snap.confidence,
                    rms: snap.rms
                )
            }
        }

        let package = ExportPackage(
            fluxchiVersion: Flux.App.version,
            schemaVersion: Flux.App.schemaVersion,
            exportedAt: iso.string(from: Date()),
            device: DeviceInfo(
                model: UIDevice.current.model,
                systemVersion: UIDevice.current.systemVersion
            ),
            session: SessionExport(
                id: session.id.uuidString,
                startedAt: iso.string(from: session.startedAt),
                endedAt: session.endedAt.map { iso.string(from: $0) },
                title: session.title,
                source: session.sourceRaw,
                durationSec: session.duration
            ),
            segments: session.segments.map { seg in
                SegmentExport(
                    id: seg.id.uuidString,
                    label: seg.labelRaw,
                    startedAt: iso.string(from: seg.startedAt),
                    endedAt: seg.endedAt.map { iso.string(from: $0) },
                    durationSec: seg.duration,
                    snapshotCount: seg.snapshots.count
                )
            },
            snapshots: allSnapshots,
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
            }
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(package)
    }

    static func shareURL(for session: Session) throws -> URL {
        let data = try export(session: session)
        let filename = "fluxchi_\(session.id.uuidString.prefix(8)).json"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        try data.write(to: url)
        return url
    }
}
