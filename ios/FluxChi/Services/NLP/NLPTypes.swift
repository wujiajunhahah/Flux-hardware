import Foundation

// MARK: - Shared Types for NLP subsystem

/// Persona for Foundation Models instructions
enum NLPPersona {
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

/// Anomaly type detected in session data
enum NLPAnomalyType: String {
    case highTension      = "持续高紧张"
    case rapidFatigue     = "快速疲劳"
    case poorRecovery     = "恢复不佳"
    case signalInstability = "信号不稳定"
}

/// Anomaly severity level
enum NLPSeverity: String {
    case info     = "提示"
    case warning  = "注意"
    case critical = "警告"
}

/// A detected anomaly in session data
struct NLPAnomaly {
    let type: NLPAnomalyType
    let message: String
    let severity: NLPSeverity
}

/// Extracted stats for a single session
struct NLPSessionStats {
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

/// Aggregated daily stats across sessions
struct NLPDailyStats {
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

/// Weekly trend analysis
struct NLPWeeklyTrend {
    let staminaTrend: String       // "上升" / "下降" / "稳定"
    let bestTimeSlot: String?      // "上午" 等
    let avgDailyMinutes: Int
    let fatigueAccumulation: String // "累积" / "稳定" / "改善"
    let summary: String            // 1-2 句趋势洞察
}

/// Context for coach follow-up questions
struct NLPCoachContext {
    let todaySessions: [Session]
    let dailyInsight: String?
    let anomalies: [NLPAnomaly]
    let weeklyTrend: NLPWeeklyTrend?
}
