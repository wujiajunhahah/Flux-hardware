import Foundation
import SwiftUI

#if canImport(FoundationModels)
import FoundationModels
#endif

@available(iOS 26.0, *)
final class NLPSummaryEngine {

    static let shared = NLPSummaryEngine()

    /// 检查端侧模型是否可用
    var isAvailable: Bool {
        #if canImport(FoundationModels)
        return SystemLanguageModel.default.isAvailable
        #else
        return false
        #endif
    }

    // MARK: - Public API

    /// 为 session 生成 AI 摘要（优先端侧 LLM，降级为模板）
    func generateSummary(for session: Session) async -> String {
        let stats = extractStats(session)
        let prompt = buildPrompt(stats)

        if let nlpResult = await tryFoundationModels(prompt: prompt) {
            return nlpResult
        }
        return fallbackSummary(stats)
    }

    /// 为 session 生成 AI 建议（单独调用，用于 SessionSummarySheet）
    func generateAdvice(for session: Session) async -> String? {
        let stats = extractStats(session)
        let prompt = buildAdvicePrompt(stats)
        return await tryFoundationModels(prompt: prompt)
    }

    // MARK: - Daily Insight (多 session 聚合总结)

    struct DailyStats {
        let sessionCount: Int
        let totalMinutes: Int
        let avgStamina: Double
        let bestStamina: Double
        let worstStamina: Double
        let totalPeakFocusMinutes: Double
        let avgFatigue: Double
        let avgTension: Double
        let avgConsistency: Double
        let timeDistribution: [String: Int]   // "上午": 2, "下午": 1
        let trend: String                     // "上升" / "下降" / "稳定"
        let unfeedbackCount: Int
    }

    /// 为今日所有 session 生成每日洞察（1-2 句话，像教练发的微信）
    func generateDailyInsight(sessions: [Session]) async -> String {
        guard !sessions.isEmpty else { return "今天还没有专注记录，开始第一段吧。" }

        let stats = extractDailyStats(sessions)
        let prompt = buildDailyPrompt(stats)

        if let nlpResult = await tryFoundationModels(prompt: prompt, persona: .dailyCoach) {
            return nlpResult
        }
        return fallbackDailyInsight(stats)
    }

    private func extractDailyStats(_ sessions: [Session]) -> DailyStats {
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

        return DailyStats(
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

    private func buildDailyPrompt(_ s: DailyStats) -> String {
        let timeSlots = s.timeDistribution.map { "\($0.key) \($0.value) 次" }.joined(separator: "、")
        return """
        请用 1-2 句话总结我今天的身体状态，像朋友发微信一样自然：

        今日数据：
        - \(s.sessionCount) 段专注，共 \(s.totalMinutes) 分钟
        - 平均续航 \(Int(s.avgStamina))，最佳 \(Int(s.bestStamina))，最低 \(Int(s.worstStamina))
        - 累计高效 \(Int(s.totalPeakFocusMinutes)) 分钟
        - 疲劳 \(String(format: "%.0f%%", s.avgFatigue * 100))，紧张 \(String(format: "%.0f%%", s.avgTension * 100))，一致性 \(String(format: "%.0f%%", s.avgConsistency * 100))
        - 时段分布：\(timeSlots)
        - 续航趋势：\(s.trend)

        要求：最多 2 句话，要有洞察力，给一个具体建议。不要用 bullet point。
        """
    }

    private func fallbackDailyInsight(_ s: DailyStats) -> String {
        var parts: [String] = []

        // 开场：基于续航评价
        if s.avgStamina >= 75 {
            parts.append("今天状态不错，\(s.sessionCount) 段专注累计 \(s.totalMinutes) 分钟，续航稳定在 \(Int(s.avgStamina))。")
        } else if s.avgStamina >= 50 {
            parts.append("今天 \(s.sessionCount) 段专注共 \(s.totalMinutes) 分钟，续航 \(Int(s.avgStamina))，中规中矩。")
        } else {
            parts.append("今天身体信号偏弱，\(s.sessionCount) 段专注续航只有 \(Int(s.avgStamina))，记得早点休息。")
        }

        // 建议：基于趋势和疲劳
        if s.trend == "下降" {
            parts.append("下午状态明显下滑，明天试试午休 15 分钟再开始。")
        } else if s.avgFatigue > 0.5 {
            parts.append("疲劳指数偏高，睡前做做拉伸，明天会更好。")
        } else if s.avgTension > 0.4 {
            parts.append("肩颈紧张度较高，注意调整坐姿和屏幕高度。")
        } else if s.trend == "上升" {
            parts.append("越到后面状态越好，找到了自己的节奏。")
        } else {
            parts.append("保持这个节奏，身体在适应中。")
        }

        return parts.joined(separator: "")
    }

    // MARK: - Foundation Models (on-device LLM)

    enum Persona {
        case sessionCoach   // 单次 session 总结
        case dailyCoach     // 每日洞察

        var instructions: String {
            switch self {
            case .sessionCoach:
                return """
                你是 FocuX 的专注力教练 —— 一个基于 EMG 生物信号的智能健康伴侣。
                你的角色是温暖、专业、有洞察力的伙伴，像一个了解身体数据的好朋友。

                风格要求：
                - 用中文回复，语气温和但直接，像朋友聊天
                - 3-5 句话，不要 bullet point，直接写段落
                - 基于 EMG 数据给出有洞察力的观察，而不是泛泛而谈
                - 适当使用鼓励，但要真诚，不要过度夸张
                - 如果数据显示疲劳，要坦诚指出并给具体建议
                - 不要提到"EMG"这个术语，用"身体信号"或"肌肉数据"代替
                """
            case .dailyCoach:
                return """
                你是 FocuX 身体教练 —— 用户的每日健康伙伴。
                你了解用户一整天的身体信号数据，像一个贴心的私人教练在晚上发一条微信总结。

                角色特质：
                - 温暖但有专业洞察力，不是客服机器人
                - 说话像朋友，不用"您"，用"你"
                - 观察数据趋势（上升/下降/稳定），而非罗列数字
                - 给出的建议必须具体可执行（比如"午休 15 分钟"而非"注意休息"）
                - 偶尔幽默，但不刻意
                - 不要提到"EMG"，用"身体信号"或"身体数据"代替

                格式要求：
                - 最多 2 句话，简洁有力
                - 不要 bullet point，写成自然的句子
                - 第一句总结状态，第二句给建议
                """
            }
        }
    }

    private func tryFoundationModels(prompt: String, persona: Persona = .sessionCoach) async -> String? {
        #if canImport(FoundationModels)
        guard SystemLanguageModel.default.isAvailable else {
            print("[NLP] Foundation Models not available on this device")
            return nil
        }
        do {
            let session = LanguageModelSession(instructions: persona.instructions)
            let response = try await session.respond(to: prompt)
            let text = response.content.trimmingCharacters(in: .whitespacesAndNewlines)
            return text.isEmpty ? nil : text
        } catch {
            print("[NLP] Foundation Models error: \(error)")
            return nil
        }
        #else
        return nil
        #endif
    }

    // MARK: - Stats Extraction

    struct SessionStats {
        let totalMinutes: Int
        let workMinutes: Int
        let restMinutes: Int
        let avgStamina: Double
        let minStamina: Double
        let maxStamina: Double
        let segmentCount: Int
        let peakFocusMinutes: Double
        let declinePointMinutes: Int?
        let staminaDelta: Double
        let activityBreakdown: [String: Int]
        let staminaCurve: [Double]
        let timeOfDay: String
        let consistencyAvg: Double
        let tensionAvg: Double
        let fatigueAvg: Double
    }

    func extractStats(_ session: Session) -> SessionStats {
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

        return SessionStats(
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

    private func longestStreak(_ values: [Double], above threshold: Double, interval: TimeInterval) -> TimeInterval {
        var maxStreak = 0, current = 0
        for v in values {
            if v >= threshold { current += 1; maxStreak = max(maxStreak, current) }
            else { current = 0 }
        }
        return Double(maxStreak) * interval
    }

    // MARK: - Prompt Builder

    private func buildPrompt(_ s: SessionStats) -> String {
        """
        请根据以下专注数据为我写一段个性化总结：

        时段: \(s.timeOfDay)
        总时长: \(s.totalMinutes) 分钟（工作 \(s.workMinutes) 分钟，休息 \(s.restMinutes) 分钟）
        续航值: 均值 \(Int(s.avgStamina))，最低 \(Int(s.minStamina))，最高 \(Int(s.maxStamina))，变化 \(s.staminaDelta > 0 ? "+" : "")\(Int(s.staminaDelta))
        最长连续高效: \(Int(s.peakFocusMinutes)) 分钟
        肌肉一致性: \(String(format: "%.0f%%", s.consistencyAvg * 100))
        紧张度: \(String(format: "%.0f%%", s.tensionAvg * 100))
        疲劳度: \(String(format: "%.0f%%", s.fatigueAvg * 100))
        分段数: \(s.segmentCount)
        \(s.declinePointMinutes.map { "约第 \($0) 分钟后开始下降" } ?? "未出现明显下降")

        请给出总结和一个具体的、可操作的建议。
        """
    }

    private func buildAdvicePrompt(_ s: SessionStats) -> String {
        """
        基于这次专注数据，给我一条简短的建议（1-2 句话）：

        续航均值: \(Int(s.avgStamina))，疲劳度: \(String(format: "%.0f%%", s.fatigueAvg * 100))，\
        紧张度: \(String(format: "%.0f%%", s.tensionAvg * 100))，\
        最长高效: \(Int(s.peakFocusMinutes)) 分钟，总时长: \(s.totalMinutes) 分钟。
        \(s.declinePointMinutes.map { "第 \($0) 分钟后开始下降。" } ?? "")

        请给一条具体、可执行的身体建议（比如拉伸、喝水、休息时长等）。
        """
    }

    // MARK: - Fallback (Template)

    func fallbackSummary(_ s: SessionStats) -> String {
        var parts: [String] = []

        let emoji: String
        if s.avgStamina >= 80 { emoji = "状态出色" }
        else if s.avgStamina >= 60 { emoji = "表现不错" }
        else if s.avgStamina >= 40 { emoji = "还有提升空间" }
        else { emoji = "身体在发信号" }

        parts.append("\(s.timeOfDay)的这段 \(s.totalMinutes) 分钟，\(emoji)。")

        if s.avgStamina >= 70 {
            parts.append("平均专注度达到 \(Int(s.avgStamina))，肌肉信号稳定，说明你进入了不错的心流状态。")
        } else if s.avgStamina >= 45 {
            parts.append("平均专注度 \(Int(s.avgStamina))，有波动但整体可控。")
        } else {
            parts.append("专注度偏低（\(Int(s.avgStamina))），身体信号显示肌肉已经在反复发出疲劳信号了。")
        }

        if s.peakFocusMinutes > 5 {
            parts.append("你最长连续保持高效 \(Int(s.peakFocusMinutes)) 分钟，这是你的「黄金窗口」。")
        }

        if let dp = s.declinePointMinutes, dp > 3 {
            parts.append("大约第 \(dp) 分钟开始出现下降，下次可以在这之前主动休息 5 分钟。")
        }

        if s.fatigueAvg > 0.6 {
            parts.append("疲劳指数较高，建议站起来活动一下，做几次深呼吸。")
        } else if s.tensionAvg > 0.4 {
            parts.append("肩颈区域紧张度偏高，试试放松双肩、转转脖子。")
        } else if s.consistencyAvg > 0.7 {
            parts.append("动作一致性很好，保持这个节奏。")
        }

        return parts.joined(separator: "")
    }
}
