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

    /// 诊断信息（用于调试 Foundation Models 降级原因）
    var diagnosticInfo: String {
        #if canImport(FoundationModels)
        let model = SystemLanguageModel.default
        if model.isAvailable {
            return "Foundation Models 可用 ✓"
        } else {
            return "Foundation Models 不可用 — 请检查: 1) 设备是否支持 Apple Intelligence  2) 设置 > Apple Intelligence 是否已开启  3) 模型是否已下载完成"
        }
        #else
        return "FoundationModels 框架未编入 — 需要 Xcode 26+ 编译"
        #endif
    }

    /// 预热模型（在 App 启动时调用，提前加载资源）
    func prewarm() {
        #if canImport(FoundationModels)
        Task {
            let session = LanguageModelSession()
            session.prewarm()
            print("[NLP] Foundation Models prewarm 已请求")
        }
        #endif
    }

    // MARK: - Public API

    /// 为 session 生成 AI 摘要（优先端侧 LLM，降级为模板）
    func generateSummary(for session: Session) async -> String {
        let stats = extractStats(session)
        let anomalies = detectAnomalies(for: session)
        let prompt = buildPrompt(stats, anomalies: anomalies)

        if let nlpResult = await tryFoundationModels(prompt: prompt) {
            return nlpResult
        }
        return fallbackSummary(stats, anomalies: anomalies)
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
    func generateDailyInsight(sessions: [Session], allRecentSessions: [Session] = []) async -> String {
        guard !sessions.isEmpty else { return "今天还没有专注记录，开始第一段吧。" }

        let stats = extractDailyStats(sessions)
        let anomalies = detectDailyAnomalies(sessions: sessions)
        let weeklyTrend = allRecentSessions.count >= 2 ? generateWeeklyTrend(sessions: allRecentSessions) : nil
        let prompt = buildDailyPrompt(stats, anomalies: anomalies, weeklyTrend: weeklyTrend)

        if let nlpResult = await tryFoundationModels(prompt: prompt, persona: .dailyCoach) {
            return nlpResult
        }
        return fallbackDailyInsight(stats, anomalies: anomalies)
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

    private func buildDailyPrompt(_ s: DailyStats, anomalies: [Anomaly] = [], weeklyTrend: WeeklyTrend? = nil) -> String {
        let timeSlots = s.timeDistribution.map { "\($0.key) \($0.value) 次" }.joined(separator: "、")

        var anomalySection = ""
        if !anomalies.isEmpty {
            let lines = anomalies.map { "- [\($0.severity.rawValue)] \($0.type.rawValue): \($0.message)" }
            anomalySection = "\n异常检测:\n\(lines.joined(separator: "\n"))\n"
        }

        var trendSection = ""
        if let trend = weeklyTrend {
            trendSection = """
            \n7 天趋势:
            - 续航趋势: \(trend.staminaTrend)
            - 日均专注: \(trend.avgDailyMinutes) 分钟
            - 最佳时段: \(trend.bestTimeSlot ?? "暂无")
            - 疲劳趋势: \(trend.fatigueAccumulation)
            """
        }

        return """
        请用 1-2 句话总结我今天的身体状态，像朋友发微信一样自然：

        今日数据：
        - \(s.sessionCount) 段专注，共 \(s.totalMinutes) 分钟
        - 平均续航 \(Int(s.avgStamina))，最佳 \(Int(s.bestStamina))，最低 \(Int(s.worstStamina))
        - 累计高效 \(Int(s.totalPeakFocusMinutes)) 分钟
        - 疲劳 \(String(format: "%.0f%%", s.avgFatigue * 100))，紧张 \(String(format: "%.0f%%", s.avgTension * 100))，一致性 \(String(format: "%.0f%%", s.avgConsistency * 100))
        - 时段分布：\(timeSlots)
        - 续航趋势：\(s.trend)
        \(anomalySection)\(trendSection)
        要求：最多 2 句话，要有洞察力，给一个具体建议。不要用 bullet point。\(anomalies.isEmpty ? "" : "请特别关注异常问题。")
        """
    }

    private func fallbackDailyInsight(_ s: DailyStats, anomalies: [Anomaly] = []) -> String {
        let hour = Calendar.current.component(.hour, from: Date())
        let greeting = hour < 12 ? "早上好" : hour < 18 ? "下午好" : "辛苦了"

        // 多样化开场（基于数据特征选择不同角度）
        var candidates: [String] = []

        if s.avgStamina >= 75 {
            candidates = [
                "\(greeting)，今天 \(s.sessionCount) 段专注、续航 \(Int(s.avgStamina))，身体状态很在线。",
                "今天节奏掌握得很好，\(s.totalMinutes) 分钟专注、平均续航 \(Int(s.avgStamina))，身体信号很稳。",
                "今天身体给了很正的反馈，\(s.sessionCount) 段专注续航都在高位，保持这个感觉。"
            ]
        } else if s.avgStamina >= 50 {
            candidates = [
                "\(greeting)，今天 \(s.sessionCount) 段专注共 \(s.totalMinutes) 分钟，续航 \(Int(s.avgStamina))，有波动但整体还行。",
                "今天的身体数据中规中矩（续航 \(Int(s.avgStamina))），\(s.trend == "上升" ? "好消息是越往后状态越好" : "试试在疲劳前就主动休息")。",
                "\(s.totalMinutes) 分钟的专注不算少，续航 \(Int(s.avgStamina)) 说明还有提升空间——关键在休息节奏。"
            ]
        } else {
            candidates = [
                "\(greeting)，今天身体信号偏弱，续航只有 \(Int(s.avgStamina))。没关系，状态有起伏是正常的。",
                "今天不是最佳状态（续航 \(Int(s.avgStamina))），身体可能需要更多恢复时间。",
                "续航 \(Int(s.avgStamina)) 偏低，身体在告诉你需要减速。早点休息，明天再来。"
            ]
        }

        var result = candidates.randomElement()!

        // 追加具体洞察（优先级：异常 > 趋势 > 维度分析）
        let critical = anomalies.filter { $0.severity == .critical }
        let warnings = anomalies.filter { $0.severity == .warning }

        if let first = critical.first {
            result += first.message + "，建议重点关注。"
        } else if let first = warnings.first {
            result += first.message + "。"
        } else if s.trend == "下降" && s.sessionCount >= 2 {
            let tips = [
                "后半段明显下滑，明天试试把重要任务放在上午。",
                "状态越往后越疲，下次在感觉还行的时候就休息一下。"
            ]
            result += tips.randomElement()!
        } else if s.avgFatigue > 0.5 {
            result += "疲劳指数 \(Int(s.avgFatigue * 100))% 偏高，睡前拉伸 10 分钟会有帮助。"
        } else if s.avgTension > 0.4 {
            result += "紧张度 \(Int(s.avgTension * 100))% 偏高，注意肩颈放松和坐姿调整。"
        } else if s.trend == "上升" {
            result += "而且越到后面状态越好，说明你找到了自己的节奏。"
        } else if s.totalPeakFocusMinutes >= 20 {
            result += "高效时段累计 \(Int(s.totalPeakFocusMinutes)) 分钟，质量不错。"
        } else if let bestSlot = s.timeDistribution.max(by: { $0.value < $1.value })?.key {
            result += "\(bestSlot)是你今天的最佳时段，可以把重要事情安排在这个时间。"
        }

        return result
    }

    /// 无 session 时的教练洞察（基于时间和历史）
    func generateEmptyDayInsight(recentSessions: [Session] = []) -> String {
        let hour = Calendar.current.component(.hour, from: Date())

        if !recentSessions.isEmpty {
            let avgVals = recentSessions.compactMap(\.avgStamina)
            let recentAvg = avgVals.isEmpty ? 0 : Int(avgVals.reduce(0, +) / Double(avgVals.count))
            let totalDays = Set(recentSessions.map { Calendar.current.startOfDay(for: $0.startedAt) }).count

            if hour < 12 {
                return "早上好！最近 \(totalDays) 天平均续航 \(recentAvg)，\(recentAvg >= 65 ? "状态不错，今天继续保持" : "今天试试每 25 分钟休息一次")。连上设备开始第一段专注吧。"
            } else if hour < 18 {
                return "下午好，今天还没有记录。你最近 \(totalDays) 天的数据显示\(recentAvg >= 65 ? "状态稳定" : "还有提升空间")，找个时间段专注一下？"
            } else {
                return "今天还没有专注记录。\(recentAvg >= 65 ? "你最近状态不错，休息一晚明天继续" : "没关系，适当休息也是提升的一部分")。"
            }
        }

        // 完全没有历史数据
        if hour < 12 {
            return "早上好！连上设备开始你的第一段专注记录，教练会根据你的身体数据给出个性化建议。"
        } else if hour < 18 {
            return "开始第一段专注记录吧，教练需要你的身体数据才能给出有针对性的建议。"
        } else {
            return "晚上好，还没有专注记录。明天一早连上设备，让教练帮你找到最佳工作节奏。"
        }
    }

    // MARK: - Weekly Trend (B2)

    struct WeeklyTrend {
        let staminaTrend: String       // "上升" / "下降" / "稳定"
        let bestTimeSlot: String?      // "上午" 等
        let avgDailyMinutes: Int
        let fatigueAccumulation: String // "累积" / "稳定" / "改善"
        let summary: String            // 1-2 句趋势洞察
    }

    /// 分析最近 7 天的趋势
    func generateWeeklyTrend(sessions: [Session]) -> WeeklyTrend {
        let calendar = Calendar.current
        let now = Date()
        let weekAgo = calendar.date(byAdding: .day, value: -7, to: now)!

        let weekSessions = sessions.filter { $0.startedAt >= weekAgo }
        guard weekSessions.count >= 2 else {
            return WeeklyTrend(
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

        return WeeklyTrend(
            staminaTrend: staminaTrend,
            bestTimeSlot: bestSlot,
            avgDailyMinutes: avgDailyMin,
            fatigueAccumulation: fatigueAccum,
            summary: summary
        )
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
        let model = SystemLanguageModel.default
        guard model.isAvailable else {
            print("[NLP] Foundation Models 不可用 — isAvailable=false")
            print("[NLP] 诊断: \(diagnosticInfo)")
            return nil
        }

        do {
            let inst = persona.instructions
            let session = LanguageModelSession {
                inst
            }
            print("[NLP] Foundation Models 开始生成 (persona: \(persona))...")
            let response = try await session.respond(to: prompt)
            let text = response.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty {
                print("[NLP] Foundation Models 返回空内容，降级到模板")
                return nil
            }
            print("[NLP] Foundation Models 生成成功 (\(text.count) 字符)")
            return text
        } catch {
            print("[NLP] Foundation Models 错误: \(error.localizedDescription)")
            print("[NLP] 错误详情: \(error)")
            return nil
        }
        #else
        print("[NLP] FoundationModels 框架未编入 (canImport 失败)")
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

    // MARK: - Coach Follow-Up (B3)

    struct CoachContext {
        let todaySessions: [Session]
        let dailyInsight: String?
        let anomalies: [Anomaly]
        let weeklyTrend: WeeklyTrend?
    }

    /// 预设追问问题
    static let presetQuestions: [String] = [
        "为什么我下午续航总是下降？",
        "怎样延长高效时间？",
        "我的紧张度正常吗？"
    ]

    /// 智能追问：基于当前上下文回答用户问题
    func askFollowUp(context: CoachContext, question: String) async -> String {
        let stats = context.todaySessions.isEmpty ? nil : extractDailyStats(context.todaySessions)

        var dataSection = "暂无今日数据。"
        if let s = stats {
            dataSection = """
            今日 \(s.sessionCount) 段专注，共 \(s.totalMinutes) 分钟，平均续航 \(Int(s.avgStamina))。
            疲劳 \(String(format: "%.0f%%", s.avgFatigue * 100))，紧张 \(String(format: "%.0f%%", s.avgTension * 100))，一致性 \(String(format: "%.0f%%", s.avgConsistency * 100))。
            """
        }

        var anomalySection = ""
        if !context.anomalies.isEmpty {
            let lines = context.anomalies.map { "- \($0.type.rawValue): \($0.message)" }
            anomalySection = "\n检测到的异常:\n\(lines.joined(separator: "\n"))"
        }

        var trendSection = ""
        if let trend = context.weeklyTrend {
            trendSection = "\n7天趋势: \(trend.summary)"
        }

        var insightSection = ""
        if let insight = context.dailyInsight {
            insightSection = "\n之前的教练总结: \(insight)"
        }

        let prompt = """
        你是一个专注力教练。用户基于以下数据向你提问，请用 2-3 句话回答，要具体、有洞察力。

        用户数据:
        \(dataSection)\(anomalySection)\(trendSection)\(insightSection)

        用户问题: \(question)

        要求：像朋友聊天一样自然，给出具体可操作的建议。
        """

        if let nlpResult = await tryFoundationModels(prompt: prompt, persona: .dailyCoach) {
            return nlpResult
        }
        return fallbackFollowUp(question: question, stats: stats, anomalies: context.anomalies)
    }

    private func fallbackFollowUp(question: String, stats: DailyStats?, anomalies: [Anomaly]) -> String {
        let q = question.lowercased()

        if q.contains("下午") || q.contains("下降") || q.contains("掉") {
            if let s = stats, s.avgFatigue > 0.5 {
                return "你的疲劳指数 \(Int(s.avgFatigue * 100))% 偏高，下午下降可能是午餐后血糖波动加上前半天的累积疲劳。两个建议：1) 午餐少吃精碳水（白米饭、面条），多吃蛋白质；2) 饭后散步 10 分钟再开始工作。"
            }
            if let s = stats, s.avgTension > 0.4 {
                return "你的紧张度偏高（\(Int(s.avgTension * 100))%），下午下降可能和持续的肩颈紧绷有关。试试每小时做一次 2 分钟的肩部放松——耸肩 5 秒、放下，重复 5 次。"
            }
            return "下午续航下降是最常见的模式，主要原因是午餐后的血糖波动和上午的疲劳累积。最有效的对策：午饭后走 10 分钟，下午第一个番茄钟从 15 分钟开始，逐渐加到 25 分钟。"
        }

        if q.contains("延长") || q.contains("高效") || q.contains("时间") || q.contains("更长") {
            if let s = stats, s.totalPeakFocusMinutes > 0 {
                let current = Int(s.totalPeakFocusMinutes)
                let target = min(current + 10, 45)
                return "你目前最长高效时段 \(current) 分钟。要延长到 \(target) 分钟，关键不是「忍着不休息」，而是在第 \(max(current - 5, 15)) 分钟时主动站起来活动 2 分钟——这种微休息能帮你把高效状态延续更久。"
            }
            return "延长高效时间的核心原则是「主动休息」——在状态还好的时候就短暂休息 3-5 分钟，比连续工作到筋疲力尽再休息效果好得多。试试 25 分钟工作 + 5 分钟休息的节奏。"
        }

        if q.contains("紧张") || q.contains("tension") || q.contains("肩") || q.contains("颈") {
            let hasAnomaly = anomalies.contains { $0.type == .highTension }
            if hasAnomaly {
                return "你的紧张度确实偏高，这说明肩颈区域在长时间紧绷。三个立刻能做的事：1) 把屏幕抬高到视线平齐；2) 每 30 分钟做一次「肩部画圈」——向前转 5 圈、向后转 5 圈；3) 打字时注意放松双肩，不要耸起来。"
            }
            if let s = stats {
                let level = s.avgTension > 0.5 ? "偏高" : s.avgTension > 0.3 ? "正常偏上" : "正常范围"
                return "你的紧张度 \(Int(s.avgTension * 100))%，属于\(level)。紧张度主要反映肩颈肌肉的持续收缩程度。低于 30% 是理想状态，30-50% 需要注意姿势，超过 50% 建议立刻做放松练习。"
            }
            return "适度紧张是正常的工作状态，但持续超过 40% 就需要注意了。最简单的判断方法：如果你需要刻意才能放松肩膀，说明紧张度已经偏高。每隔 30 分钟检查一次肩膀是否耸起来。"
        }

        if q.contains("疲劳") || q.contains("累") || q.contains("fatigue") {
            if let s = stats {
                if s.avgFatigue > 0.6 {
                    return "你的疲劳指数 \(Int(s.avgFatigue * 100))% 确实偏高。疲劳不是一下子出现的，而是逐渐累积的。建议今天早点结束工作，睡前做 10 分钟拉伸。明天试试缩短每段专注的时长（20 分钟），增加休息次数。"
                } else {
                    return "疲劳指数 \(Int(s.avgFatigue * 100))% 还在可控范围。要保持低疲劳，关键是不要等到累了才休息——在感觉还行的时候就主动休息 5 分钟，这样全天的疲劳累积会明显减少。"
                }
            }
            return "疲劳的核心管理原则是「预防大于治疗」。与其等到精疲力尽再休息 30 分钟，不如每 25 分钟就休息 5 分钟。后者全天总工作量反而更高。"
        }

        if q.contains("恢复") || q.contains("休息") || q.contains("recovery") {
            let hasPoorRecovery = anomalies.contains { $0.type == .poorRecovery }
            if hasPoorRecovery {
                return "你的恢复效率确实不太够——休息后续航回升不明显。可能原因：1) 休息时间太短（至少需要 5-10 分钟）；2) 休息时还在看手机（大脑没有真正休息）。真正的休息是：站起来、看远处、做几次深呼吸。"
            }
            return "有效的休息不是刷手机，而是让身体和大脑都切换模式。最高效的恢复方式：站起来走一圈、看 20 秒远处的绿色植物、做 5 次深呼吸。这比坐在椅子上刷 15 分钟手机恢复效果好 3 倍。"
        }

        if q.contains("最佳") || q.contains("什么时候") || q.contains("时段") || q.contains("时间段") {
            if let s = stats, let bestSlot = s.timeDistribution.max(by: { $0.value < $1.value })?.key {
                return "从今天的数据看，\(bestSlot)是你状态最好的时段。一般来说，大多数人的认知高峰在上午 9-11 点和下午 3-5 点。建议把最重要、最需要创造力的任务放在你的高峰时段。"
            }
            return "每个人的最佳时段不同，但大部分人有两个高峰：上午 9-11 点和下午 3-5 点。连续记录一周后，教练就能帮你找到属于你的最佳时段。"
        }

        // 通用回答——也要有变化
        let genericAnswers = [
            "这是个好问题。从你的数据来看，建议关注两件事：1) 主动休息——每 25 分钟站起来活动 5 分钟；2) 注意姿势——肩膀放松、屏幕与视线平齐。这两个习惯坚持一周就能看到续航提升。",
            "基于你的身体数据，最重要的建议是建立固定节奏——工作 25 分钟、休息 5 分钟、每 2 小时一次长休息 15 分钟。节奏感比「坚持更久」更能提升效率。",
            "好问题。如果只能给一个建议，那就是「主动休息」——不要等身体发出疲劳信号才休息，在状态还好的时候就主动停下来 5 分钟。这一个习惯就能带来明显改变。"
        ]
        return genericAnswers.randomElement()!
    }

    // MARK: - Anomaly Detection (B1)

    enum AnomalyType: String {
        case highTension      = "持续高紧张"
        case rapidFatigue     = "快速疲劳"
        case poorRecovery     = "恢复不佳"
        case signalInstability = "信号不稳定"
    }

    enum Severity: String {
        case info     = "提示"
        case warning  = "注意"
        case critical = "警告"
    }

    struct Anomaly {
        let type: AnomalyType
        let message: String
        let severity: Severity
    }

    /// 检测 session 中的异常模式
    func detectAnomalies(for session: Session) -> [Anomaly] {
        let snapshots = session.segments.flatMap { $0.snapshots }.sorted { $0.timestamp < $1.timestamp }
        guard !snapshots.isEmpty else { return [] }

        var anomalies: [Anomaly] = []

        // 1. 持续高紧张：tension > 0.6 超过 60% 的采样点
        let highTensionCount = snapshots.filter { $0.tension > 0.6 }.count
        let tensionRatio = Double(highTensionCount) / Double(snapshots.count)
        if tensionRatio > 0.6 {
            anomalies.append(Anomaly(
                type: .highTension,
                message: "本次 \(Int(tensionRatio * 100))% 的时间紧张度偏高，肩颈可能过度紧绷",
                severity: tensionRatio > 0.8 ? .critical : .warning
            ))
        }

        // 2. 快速疲劳：前 1/3 时间内 fatigue 就超过 0.5
        let earlyCount = max(snapshots.count / 3, 1)
        let earlyFatigue = snapshots.prefix(earlyCount).map(\.fatigue).reduce(0, +) / Double(earlyCount)
        if earlyFatigue > 0.5 {
            anomalies.append(Anomaly(
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
                    anomalies.append(Anomaly(
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
            anomalies.append(Anomaly(
                type: .signalInstability,
                message: "肌肉信号波动较大（\(Int(unstableRatio * 100))% 时间不稳定），检查设备佩戴是否贴合",
                severity: unstableRatio > 0.6 ? .warning : .info
            ))
        }

        return anomalies
    }

    /// 检测每日聚合异常
    func detectDailyAnomalies(sessions: [Session]) -> [Anomaly] {
        var anomalies: [Anomaly] = []
        for session in sessions {
            anomalies.append(contentsOf: detectAnomalies(for: session))
        }
        // 去重：同类型只保留最严重的
        var seen: [AnomalyType: Anomaly] = [:]
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

    private func severityRank(_ s: Severity) -> Int {
        switch s {
        case .info: return 0
        case .warning: return 1
        case .critical: return 2
        }
    }

    // MARK: - Prompt Builder

    private func buildPrompt(_ s: SessionStats, anomalies: [Anomaly] = []) -> String {
        var anomalySection = ""
        if !anomalies.isEmpty {
            let lines = anomalies.map { "- [\($0.severity.rawValue)] \($0.type.rawValue): \($0.message)" }
            anomalySection = "\n异常检测:\n\(lines.joined(separator: "\n"))\n"
        }

        return """
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
        \(anomalySection)
        请给出总结和一个具体的、可操作的建议。\(anomalies.isEmpty ? "" : "请特别关注异常检测中发现的问题。")
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

    func fallbackSummary(_ s: SessionStats, anomalies: [Anomaly] = []) -> String {
        var parts: [String] = []

        // 多样化开场（每个等级 3 个变体，随机选一个）
        if s.avgStamina >= 80 {
            parts.append([
                "\(s.timeOfDay)这 \(s.totalMinutes) 分钟状态出色，平均续航 \(Int(s.avgStamina))，身体信号非常稳。",
                "这段 \(s.totalMinutes) 分钟的专注质量很高——续航 \(Int(s.avgStamina))，肌肉数据显示你进入了很好的心流状态。",
                "\(s.totalMinutes) 分钟，续航 \(Int(s.avgStamina))，\(s.timeOfDay)的这段是今天的高光。"
            ].randomElement()!)
        } else if s.avgStamina >= 60 {
            parts.append([
                "\(s.timeOfDay)这 \(s.totalMinutes) 分钟表现不错，续航 \(Int(s.avgStamina))，身体在配合你的节奏。",
                "\(s.totalMinutes) 分钟的专注，续航 \(Int(s.avgStamina))，整体稳定，有进一步提升的空间。",
                "这段专注 \(s.totalMinutes) 分钟，续航 \(Int(s.avgStamina))——不算惊艳但很扎实。"
            ].randomElement()!)
        } else if s.avgStamina >= 40 {
            parts.append([
                "\(s.timeOfDay)这 \(s.totalMinutes) 分钟，续航 \(Int(s.avgStamina))，有波动但坚持下来了。",
                "\(s.totalMinutes) 分钟专注，续航 \(Int(s.avgStamina))，身体信号说还有提升空间——关键在休息节奏。",
                "续航 \(Int(s.avgStamina))，不是最佳状态，但\(s.totalMinutes) 分钟的坚持本身就值得认可。"
            ].randomElement()!)
        } else {
            parts.append([
                "\(s.timeOfDay)这段身体信号偏弱，续航 \(Int(s.avgStamina))，肌肉数据在反复发出疲劳信号。",
                "续航只有 \(Int(s.avgStamina))，身体今天确实在告诉你需要慢下来。",
                "这 \(s.totalMinutes) 分钟续航 \(Int(s.avgStamina))，身体需要更多恢复，不要勉强。"
            ].randomElement()!)
        }

        // 黄金窗口洞察
        if s.peakFocusMinutes >= 10 {
            parts.append("连续高效了 \(Int(s.peakFocusMinutes)) 分钟，这是你的「黄金窗口」——下次可以用它来规划番茄钟时长。")
        } else if s.peakFocusMinutes >= 5 {
            parts.append("最长高效 \(Int(s.peakFocusMinutes)) 分钟，可以试着逐步延长到 20 分钟以上。")
        }

        // 下降点洞察
        if let dp = s.declinePointMinutes, dp > 3 {
            parts.append([
                "大约第 \(dp) 分钟开始走下坡，下次在这之前主动休息 5 分钟。",
                "第 \(dp) 分钟后续航开始下降——这是你的「转折点」，记住它。"
            ].randomElement()!)
        }

        // 异常或维度分析（每次只选一个最相关的）
        let critical = anomalies.filter { $0.severity == .critical }
        let warnings = anomalies.filter { $0.severity == .warning }

        if let first = critical.first {
            parts.append(first.message + "，需要重点注意。")
        } else if let first = warnings.first {
            parts.append(first.message + "。")
        } else if s.fatigueAvg > 0.6 {
            parts.append([
                "疲劳指数 \(Int(s.fatigueAvg * 100))% 偏高，站起来活动一下，做几次深呼吸。",
                "身体疲劳度达到 \(Int(s.fatigueAvg * 100))%，喝杯水、伸个懒腰再继续。"
            ].randomElement()!)
        } else if s.tensionAvg > 0.4 {
            parts.append("紧张度 \(Int(s.tensionAvg * 100))% 偏高，检查一下坐姿——屏幕是否在视线正前方、肩膀有没有耸起来。")
        } else if s.consistencyAvg > 0.7 {
            parts.append("动作一致性 \(Int(s.consistencyAvg * 100))% 非常稳定，说明你的工作姿势很到位。")
        } else if s.staminaDelta > 10 {
            parts.append("续航整体呈上升趋势（+\(Int(s.staminaDelta))），越做越进入状态。")
        } else if s.staminaDelta < -15 {
            parts.append("续航下降了 \(Int(abs(s.staminaDelta))) 点，下次可以缩短时长、增加休息频率。")
        }

        return parts.joined(separator: "")
    }
}
