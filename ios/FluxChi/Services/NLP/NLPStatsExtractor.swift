import Foundation

/// Stats extraction from Session data — single session + daily aggregation + weekly trend.
@available(iOS 26.0, *)
enum NLPStatsExtractor {

    // MARK: - Session Stats

    static func extractSessionStats(_ session: Session) -> NLPSessionStats {
        let allSnapshots = session.segments
            .flatMap { $0.snapshots }
            .sorted { $0.timestamp < $1.timestamp }

        let staminaValues = allSnapshots.map(\.stamina)
        let avg = staminaValues.isEmpty ? 0 : staminaValues.reduce(0, +) / Double(staminaValues.count)
        let minVal = staminaValues.min() ?? 0
        let maxVal = staminaValues.max() ?? 100

        let totalSec = session.duration
        let workSec = session.segments.filter { $0.label == .work }.reduce(0.0) { $0 + $1.duration }

        let actBreak = Dictionary(grouping: allSnapshots, by: \.activity).mapValues(\.count)

        let interval = TimeInterval(Flux.App.snapshotIntervalMs) / 1000.0
        let peakMin = longestStreak(staminaValues, above: 60, interval: interval) / 60

        var declineMin: Int?
        var wasAbove = false
        for snap in allSnapshots {
            if snap.stamina >= 60 { wasAbove = true }
            else if wasAbove {
                declineMin = Int(snap.timestamp.timeIntervalSince(session.startedAt) / 60)
                break
            }
        }

        let hour = Calendar.current.component(.hour, from: session.startedAt)
        let timeOfDay: String
        switch hour {
        case 5..<9:   timeOfDay = "清晨"
        case 9..<12:  timeOfDay = "上午"
        case 12..<14: timeOfDay = "午间"
        case 14..<18: timeOfDay = "下午"
        case 18..<22: timeOfDay = "晚间"
        default:      timeOfDay = "深夜"
        }

        let consistencyAvg = allSnapshots.isEmpty ? 0 : allSnapshots.map(\.consistency).reduce(0, +) / Double(allSnapshots.count)
        let tensionAvg = allSnapshots.isEmpty ? 0 : allSnapshots.map(\.tension).reduce(0, +) / Double(allSnapshots.count)
        let fatigueAvg = allSnapshots.isEmpty ? 0 : allSnapshots.map(\.fatigue).reduce(0, +) / Double(allSnapshots.count)

        let delta = (staminaValues.last ?? 0) - (staminaValues.first ?? 0)

        let curve = SummaryEngine.generate(for: session).staminaCurve

        return NLPSessionStats(
            totalMinutes: Int(totalSec / 60),
            workMinutes: Int(workSec / 60),
            restMinutes: Int((totalSec - workSec) / 60),
            avgStamina: avg,
            minStamina: minVal,
            maxStamina: maxVal,
            segmentCount: session.segments.count,
            peakFocusMinutes: peakMin,
            declinePointMinutes: declineMin,
            staminaDelta: delta,
            activityBreakdown: actBreak,
            staminaCurve: curve,
            timeOfDay: timeOfDay,
            consistencyAvg: consistencyAvg,
            tensionAvg: tensionAvg,
            fatigueAvg: fatigueAvg
        )
    }

    // MARK: - Daily Stats

    static func extractDailyStats(_ sessions: [Session]) -> NLPDailyStats {
        let totalMin = Int(sessions.reduce(0) { $0 + $1.duration } / 60)
        let avgVals = sessions.compactMap(\.avgStamina)
        let avg = avgVals.isEmpty ? 0.0 : avgVals.reduce(0, +) / Double(avgVals.count)
        let best = avgVals.max() ?? 0
        let worst = avgVals.min() ?? 0

        // 聚合所有 snapshot
        let allSnapshots = sessions.flatMap { $0.segments.flatMap { $0.snapshots } }
        let fatigueAvg = allSnapshots.isEmpty ? 0 : allSnapshots.map(\.fatigue).reduce(0, +) / Double(allSnapshots.count)
        let tensionAvg = allSnapshots.isEmpty ? 0 : allSnapshots.map(\.tension).reduce(0, +) / Double(allSnapshots.count)
        let consistencyAvg = allSnapshots.isEmpty ? 0 : allSnapshots.map(\.consistency).reduce(0, +) / Double(allSnapshots.count)

        // 时段分布
        var timeDist: [String: Int] = [:]
        for s in sessions {
            let hour = Calendar.current.component(.hour, from: s.startedAt)
            let slot: String
            switch hour {
            case 5..<9:   slot = "清晨"
            case 9..<12:  slot = "上午"
            case 12..<14: slot = "午间"
            case 14..<18: slot = "下午"
            case 18..<22: slot = "晚间"
            default:      slot = "深夜"
            }
            timeDist[slot, default: 0] += 1
        }

        // 趋势：比较前后两半 session 的平均续航
        let trend: String
        if avgVals.count >= 2 {
            let mid = avgVals.count / 2
            let firstHalf = avgVals.prefix(mid).reduce(0, +) / Double(mid)
            let secondHalf = avgVals.suffix(from: mid).reduce(0, +) / Double(avgVals.count - mid)
            let delta = secondHalf - firstHalf
            if delta > 5 { trend = "上升" }
            else if delta < -5 { trend = "下降" }
            else { trend = "稳定" }
        } else {
            trend = "稳定"
        }

        // 累计高效时长
        let interval = TimeInterval(Flux.App.snapshotIntervalMs) / 1000.0
        let peakMin = sessions.reduce(0.0) { total, session in
            let vals = session.segments.flatMap { $0.snapshots }.sorted { $0.timestamp < $1.timestamp }.map(\.stamina)
            return total + longestStreak(vals, above: 60, interval: interval) / 60
        }

        let unfeedback = sessions.filter { $0.feedback == nil }.count

        return NLPDailyStats(
            sessionCount: sessions.count,
            totalMinutes: totalMin,
            avgStamina: avg,
            bestStamina: best,
            worstStamina: worst,
            totalPeakFocusMinutes: peakMin,
            avgFatigue: fatigueAvg,
            avgTension: tensionAvg,
            avgConsistency: consistencyAvg,
            timeDistribution: timeDist,
            trend: trend,
            unfeedbackCount: unfeedback
        )
    }

    // MARK: - Weekly Trend

    static func generateWeeklyTrend(sessions: [Session]) -> NLPWeeklyTrend {
        let calendar = Calendar.current
        let now = Date()
        let weekAgo = calendar.date(byAdding: .day, value: -7, to: now)!

        let weekSessions = sessions.filter { $0.startedAt >= weekAgo }
        guard weekSessions.count >= 2 else {
            return NLPWeeklyTrend(
                staminaTrend: "稳定",
                bestTimeSlot: nil,
                avgDailyMinutes: 0,
                fatigueAccumulation: "稳定",
                summary: "数据不足，继续记录以获得趋势分析。"
            )
        }

        // 按天分组
        let grouped = Dictionary(grouping: weekSessions) { session in
            calendar.startOfDay(for: session.startedAt)
        }
        let activeDays = grouped.count
        let totalMin = Int(weekSessions.reduce(0) { $0 + $1.duration } / 60)
        let avgDailyMin = activeDays > 0 ? totalMin / activeDays : 0

        // 续航趋势：比较前半周 vs 后半周
        let sorted = weekSessions.sorted { $0.startedAt < $1.startedAt }
        let mid = sorted.count / 2
        let firstHalfAvg = sorted.prefix(mid).compactMap(\.avgStamina)
        let secondHalfAvg = sorted.suffix(from: mid).compactMap(\.avgStamina)
        let firstAvg = firstHalfAvg.isEmpty ? 0.0 : firstHalfAvg.reduce(0, +) / Double(firstHalfAvg.count)
        let secondAvg = secondHalfAvg.isEmpty ? 0.0 : secondHalfAvg.reduce(0, +) / Double(secondHalfAvg.count)
        let delta = secondAvg - firstAvg

        let staminaTrend: String
        if delta > 5 { staminaTrend = "上升" }
        else if delta < -5 { staminaTrend = "下降" }
        else { staminaTrend = "稳定" }

        // 最佳时段
        var slotScores: [String: (total: Double, count: Int)] = [:]
        for s in weekSessions {
            let hour = calendar.component(.hour, from: s.startedAt)
            let slot: String
            switch hour {
            case 5..<9:   slot = "清晨"
            case 9..<12:  slot = "上午"
            case 12..<14: slot = "午间"
            case 14..<18: slot = "下午"
            case 18..<22: slot = "晚间"
            default:      slot = "深夜"
            }
            if let avg = s.avgStamina {
                slotScores[slot, default: (0, 0)].total += avg
                slotScores[slot, default: (0, 0)].count += 1
            }
        }
        let bestSlot = slotScores.max { a, b in
            (a.value.total / Double(max(a.value.count, 1))) < (b.value.total / Double(max(b.value.count, 1)))
        }?.key

        // 疲劳积累
        let firstFatigue = sorted.prefix(mid).flatMap { $0.segments.flatMap { $0.snapshots } }.map(\.fatigue)
        let secondFatigue = sorted.suffix(from: mid).flatMap { $0.segments.flatMap { $0.snapshots } }.map(\.fatigue)
        let firstFatigueAvg = firstFatigue.isEmpty ? 0.0 : firstFatigue.reduce(0, +) / Double(firstFatigue.count)
        let secondFatigueAvg = secondFatigue.isEmpty ? 0.0 : secondFatigue.reduce(0, +) / Double(secondFatigue.count)
        let fatigueDelta = secondFatigueAvg - firstFatigueAvg

        let fatigueAccum: String
        if fatigueDelta > 0.1 { fatigueAccum = "累积" }
        else if fatigueDelta < -0.1 { fatigueAccum = "改善" }
        else { fatigueAccum = "稳定" }

        // 生成摘要
        var summary = ""
        switch staminaTrend {
        case "上升":
            summary = "近 7 天续航呈上升趋势（+\(Int(delta))），身体在逐步适应。"
        case "下降":
            summary = "近 7 天续航有所下降（\(Int(delta))），注意休息和恢复。"
        default:
            summary = "近 7 天续航保持稳定，日均专注 \(avgDailyMin) 分钟。"
        }
        if let best = bestSlot {
            summary += "\(best)是你的最佳时段。"
        }
        if fatigueAccum == "累积" {
            summary += "疲劳有累积趋势，建议安排一天轻度活动。"
        }

        return NLPWeeklyTrend(
            staminaTrend: staminaTrend,
            bestTimeSlot: bestSlot,
            avgDailyMinutes: avgDailyMin,
            fatigueAccumulation: fatigueAccum,
            summary: summary
        )
    }

    // MARK: - Helpers

    static func longestStreak(_ values: [Double], above threshold: Double, interval: TimeInterval) -> TimeInterval {
        var maxStreak = 0, current = 0
        for v in values {
            if v >= threshold { current += 1; maxStreak = max(maxStreak, current) }
            else { current = 0 }
        }
        return Double(maxStreak) * interval
    }
}
