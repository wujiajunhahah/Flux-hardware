import Foundation

struct SessionSummary {
    let text: String
    let avgStamina: Double
    let minStamina: Double
    let maxStamina: Double
    let totalDuration: TimeInterval
    let workDuration: TimeInterval
    let restDuration: TimeInterval
    let segmentCount: Int
    let staminaCurve: [Double]
    let peakFocusMinutes: Double
    let declinePoint: TimeInterval?
    let recoveryRate: Double?
}

enum SummaryEngine {

    static func generate(for session: Session) -> SessionSummary {
        let allSnapshots = session.segments
            .flatMap { $0.snapshots }
            .sorted { $0.timestamp < $1.timestamp }

        guard !allSnapshots.isEmpty else { return emptySummary(for: session) }

        let staminaValues = allSnapshots.map { $0.stamina }
        let avg = staminaValues.reduce(0, +) / Double(staminaValues.count)
        let minVal = staminaValues.min() ?? 0
        let maxVal = staminaValues.max() ?? 100

        let totalDuration = session.duration
        let workDuration = session.segments
            .filter { $0.label == .work }
            .reduce(0.0) { $0 + $1.duration }
        let restDuration = totalDuration - workDuration

        let curve = downsample(staminaValues, targetCount: 200)
        let interval = TimeInterval(Flux.App.snapshotIntervalMs) / 1000.0
        let peakMinutes = longestStreak(staminaValues, above: 60, intervalSec: interval) / 60
        let declinePoint = findDeclinePoint(allSnapshots, threshold: 60, sessionStart: session.startedAt)
        let recoveryRate = computeRecoveryRate(session: session)

        let text = generateText(
            duration: totalDuration, avg: avg, minVal: minVal, maxVal: maxVal,
            workDuration: workDuration, peakMinutes: peakMinutes,
            declinePoint: declinePoint, recoveryRate: recoveryRate,
            segmentCount: session.segments.count
        )

        return SessionSummary(
            text: text,
            avgStamina: avg, minStamina: minVal, maxStamina: maxVal,
            totalDuration: totalDuration,
            workDuration: workDuration, restDuration: restDuration,
            segmentCount: session.segments.count,
            staminaCurve: curve,
            peakFocusMinutes: peakMinutes,
            declinePoint: declinePoint,
            recoveryRate: recoveryRate
        )
    }

    /// Write summary data back to the Session model.
    static func apply(_ summary: SessionSummary, to session: Session) {
        session.summaryText = summary.text
        session.avgStamina = summary.avgStamina
        session.minStamina = summary.minStamina
        session.maxStamina = summary.maxStamina
        session.workDurationSec = summary.workDuration
        session.restDurationSec = summary.restDuration
        session.segmentCount = summary.segmentCount
        session.staminaCurveData = try? JSONEncoder().encode(summary.staminaCurve)
    }

    // MARK: - Text Generation

    private static func generateText(
        duration: TimeInterval, avg: Double, minVal: Double, maxVal: Double,
        workDuration: TimeInterval, peakMinutes: Double,
        declinePoint: TimeInterval?, recoveryRate: Double?,
        segmentCount: Int
    ) -> String {
        var parts: [String] = []
        let durationMin = Int(duration / 60)
        let workMin = Int(workDuration / 60)

        parts.append("本次记录共 \(durationMin) 分钟，其中工作 \(workMin) 分钟。")

        if avg >= 75 {
            parts.append("整体状态优秀，平均专注度 \(Int(avg))，保持了较高的工作效率。")
        } else if avg >= 50 {
            parts.append("整体状态良好，平均专注度 \(Int(avg))。")
        } else if avg >= 30 {
            parts.append("专注度偏低（平均 \(Int(avg))），工作效率有待提升。")
        } else {
            parts.append("本次状态较差，平均专注度仅 \(Int(avg))，建议调整作息。")
        }

        if peakMinutes > 5 {
            parts.append("最长连续高效时段为 \(Int(peakMinutes)) 分钟。")
        }

        if let dp = declinePoint {
            let dpMin = Int(dp / 60)
            if dpMin > 3 {
                parts.append("工作约 \(dpMin) 分钟后专注度开始下降，建议下次在此之前主动休息。")
            }
        }

        if let rate = recoveryRate, rate > 0 {
            if rate > 5 {
                parts.append("休息恢复速度较快，恢复能力良好。")
            } else if rate > 2 {
                parts.append("休息恢复速度适中。")
            } else {
                parts.append("恢复速度较慢，建议延长休息时间。")
            }
        }

        if segmentCount > 1 {
            parts.append("共 \(segmentCount) 个分段。")
        }

        return parts.joined(separator: "")
    }

    // MARK: - Analysis Helpers

    private static func downsample(_ values: [Double], targetCount: Int) -> [Double] {
        guard values.count > targetCount else { return values }
        let step = Double(values.count) / Double(targetCount)
        return (0..<targetCount).map { values[min(Int(Double($0) * step), values.count - 1)] }
    }

    private static func longestStreak(_ values: [Double], above threshold: Double, intervalSec: TimeInterval) -> TimeInterval {
        var maxStreak = 0
        var current = 0
        for v in values {
            if v >= threshold {
                current += 1
                maxStreak = max(maxStreak, current)
            } else {
                current = 0
            }
        }
        return Double(maxStreak) * intervalSec
    }

    private static func findDeclinePoint(_ snapshots: [FluxSnapshot], threshold: Double, sessionStart: Date) -> TimeInterval? {
        var wasAbove = false
        for snap in snapshots {
            if snap.stamina >= threshold {
                wasAbove = true
            } else if wasAbove {
                return snap.timestamp.timeIntervalSince(sessionStart)
            }
        }
        return nil
    }

    private static func computeRecoveryRate(session: Session) -> Double? {
        let restSegments = session.segments.filter { $0.label == .rest || $0.label == .pause }
        guard !restSegments.isEmpty else { return nil }

        var totalGain = 0.0
        var totalMinutes = 0.0

        for seg in restSegments {
            let snaps = seg.snapshots.sorted { $0.timestamp < $1.timestamp }
            guard let first = snaps.first, let last = snaps.last else { continue }
            let minutes = seg.duration / 60
            if minutes > 0 {
                totalGain += last.stamina - first.stamina
                totalMinutes += minutes
            }
        }

        return totalMinutes > 0 ? totalGain / totalMinutes : nil
    }

    private static func emptySummary(for session: Session) -> SessionSummary {
        SessionSummary(
            text: "本次记录未采集到数据。",
            avgStamina: 0, minStamina: 0, maxStamina: 0,
            totalDuration: session.duration,
            workDuration: 0, restDuration: 0,
            segmentCount: 0, staminaCurve: [],
            peakFocusMinutes: 0, declinePoint: nil, recoveryRate: nil
        )
    }
}
