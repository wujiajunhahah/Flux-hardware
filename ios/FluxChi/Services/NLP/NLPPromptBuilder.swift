import Foundation

/// All prompt construction for NLP generation.
@available(iOS 26.0, *)
enum NLPPromptBuilder {

    // MARK: - Session Prompts

    static func buildSessionPrompt(_ s: NLPSessionStats, anomalies: [NLPAnomaly] = []) -> String {
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

    static func buildAdvicePrompt(_ s: NLPSessionStats) -> String {
        """
        基于这次专注数据，给我一条简短的建议（1-2 句话）：

        续航均值: \(Int(s.avgStamina))，疲劳度: \(String(format: "%.0f%%", s.fatigueAvg * 100))，\
        紧张度: \(String(format: "%.0f%%", s.tensionAvg * 100))，\
        最长高效: \(Int(s.peakFocusMinutes)) 分钟，总时长: \(s.totalMinutes) 分钟。
        \(s.declinePointMinutes.map { "第 \($0) 分钟后开始下降。" } ?? "")

        请给一条具体、可执行的身体建议（比如拉伸、喝水、休息时长等）。
        """
    }

    // MARK: - Daily Prompts

    static func buildDailyPrompt(_ s: NLPDailyStats, anomalies: [NLPAnomaly] = [], weeklyTrend: NLPWeeklyTrend? = nil) -> String {
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

    // MARK: - Follow-Up Prompt

    static func buildFollowUpPrompt(context: NLPCoachContext, stats: NLPDailyStats?, question: String) -> String {
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

        return """
        你是一个专注力教练。用户基于以下数据向你提问，请用 2-3 句话回答，要具体、有洞察力。

        用户数据:
        \(dataSection)\(anomalySection)\(trendSection)\(insightSection)

        用户问题: \(question)

        要求：像朋友聊天一样自然，给出具体可操作的建议。
        """
    }
}
