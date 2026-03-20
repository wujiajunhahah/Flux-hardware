import Foundation

/// Anomaly detection in session EMG data.
@available(iOS 26.0, *)
enum NLPAnomalyDetector {

    /// Detect anomalies for a single session
    static func detectAnomalies(for session: Session) -> [NLPAnomaly] {
        let snapshots = session.segments.flatMap { $0.snapshots }.sorted { $0.timestamp < $1.timestamp }
        guard !snapshots.isEmpty else { return [] }

        var anomalies: [NLPAnomaly] = []

        // 1. 持续高紧张：tension > 0.6 超过 60% 的采样点
        let highTensionCount = snapshots.filter { $0.tension > 0.6 }.count
        let tensionRatio = Double(highTensionCount) / Double(snapshots.count)
        if tensionRatio > 0.6 {
            anomalies.append(NLPAnomaly(
                type: .highTension,
                message: "本次 \(Int(tensionRatio * 100))% 的时间紧张度偏高，肩颈可能过度紧绷",
                severity: tensionRatio > 0.8 ? .critical : .warning
            ))
        }

        // 2. 快速疲劳：前 1/3 时间内 fatigue 就超过 0.5
        let earlyCount = max(snapshots.count / 3, 1)
        let earlyFatigue = snapshots.prefix(earlyCount).map(\.fatigue).reduce(0, +) / Double(earlyCount)
        if earlyFatigue > 0.5 {
            anomalies.append(NLPAnomaly(
                type: .rapidFatigue,
                message: "开始不久就出现明显疲劳（前期疲劳度 \(Int(earlyFatigue * 100))%），可能需要更多热身",
                severity: earlyFatigue > 0.7 ? .critical : .warning
            ))
        }

        // 3. 恢复不佳：休息段后续航恢复 < 20%
        let segments = session.segments
        for i in 1..<segments.count {
            if segments[i-1].label == .rest && segments[i].label == .work {
                let restEnd = segments[i-1].snapshots.last?.stamina ?? 0
                let workStart = segments[i].snapshots.first?.stamina ?? 0
                let recovery = workStart - restEnd
                if recovery < 20 && restEnd < 60 {
                    anomalies.append(NLPAnomaly(
                        type: .poorRecovery,
                        message: "休息后续航仅恢复 \(Int(recovery)) 点，短暂休息可能不够",
                        severity: recovery < 10 ? .warning : .info
                    ))
                    break // 只报告第一次
                }
            }
        }

        // 4. 信号不稳定：consistency < 0.3 超过 40% 的采样点
        let unstableCount = snapshots.filter { $0.consistency < 0.3 }.count
        let unstableRatio = Double(unstableCount) / Double(snapshots.count)
        if unstableRatio > 0.4 {
            anomalies.append(NLPAnomaly(
                type: .signalInstability,
                message: "肌肉信号波动较大（\(Int(unstableRatio * 100))% 时间不稳定），检查设备佩戴是否贴合",
                severity: unstableRatio > 0.6 ? .warning : .info
            ))
        }

        return anomalies
    }

    /// Detect daily aggregated anomalies (dedup by type, keep most severe)
    static func detectDailyAnomalies(sessions: [Session]) -> [NLPAnomaly] {
        var anomalies: [NLPAnomaly] = []
        for session in sessions {
            anomalies.append(contentsOf: detectAnomalies(for: session))
        }
        // 去重：同类型只保留最严重的
        var seen: [NLPAnomalyType: NLPAnomaly] = [:]
        for a in anomalies {
            if let existing = seen[a.type] {
                if severityRank(a.severity) > severityRank(existing.severity) {
                    seen[a.type] = a
                }
            } else {
                seen[a.type] = a
            }
        }
        return Array(seen.values)
    }

    private static func severityRank(_ s: NLPSeverity) -> Int {
        switch s {
        case .info: return 0
        case .warning: return 1
        case .critical: return 2
        }
    }
}
