import Foundation
import SwiftUI

#if canImport(FoundationModels)
import FoundationModels
#endif

/**
 BodyInsightEngine — 体能状态解读引擎

 **定位**：镜子而非教练
 - 呈现"身体说了什么"，而非"你应该做什么"
 - 客观描述状态模式，让用户自主决策
 - Sense, not Control — 感知而非控制
 */

@available(iOS 26.0, *)
final class BodyInsightEngine {

    static let shared = BodyInsightEngine()

    // MARK: - Availability

    var isAvailable: Bool {
        #if canImport(FoundationModels)
        return SystemLanguageModel.default.isAvailable
        #else
        return false
        #endif
    }

    var diagnosticInfo: String {
        #if canImport(FoundationModels)
        let model = SystemLanguageModel.default
        if model.isAvailable {
            return "Foundation Models 可用"
        } else {
            return "Foundation Models 不可用 — 需 iOS 26+ + Apple Intelligence"
        }
        #else
        return "FoundationModels 框架未编入"
        #endif
    }

    func prewarm() {
        #if canImport(FoundationModels)
        Task {
            let session = LanguageModelSession()
            session.prewarm()
        }
        #endif
    }

    // MARK: - Public API

    /// 为 session 生成体能洞察（状态描述，非建议）
    func generateInsight(for session: Session) async -> String {
        let stats = extractStats(session)
        let patterns = detectPatterns(for: session)
        let prompt = buildInsightPrompt(stats, patterns: patterns)

        if let result = await tryFoundationModels(prompt: prompt) {
            return result
        }
        return fallbackInsight(stats, patterns: patterns)
    }

    /// 生成每日体能总结
    func generateDailySummary(sessions: [Session]) async -> String {
        guard !sessions.isEmpty else { return "今天还没有记录。" }

        let stats = extractDailyStats(sessions)
        let patterns = detectDailyPatterns(sessions: sessions)
        let prompt = buildDailyPrompt(stats, patterns: patterns)

        if let result = await tryFoundationModels(prompt: prompt) {
            return result
        }
        return fallbackDailySummary(stats, patterns: patterns)
    }

    // MARK: - Pattern Detection

    struct Pattern {
        let type: PatternType
        let description: String
        let data: [String: Any]
    }

    enum PatternType: String {
        case earlyDecline       = "早期下降"      // 前 1/3 就开始疲劳
        case stablePerformance = "稳定表现"      // 续航平稳
        case latePeak          = "后程发力"      // 后半段更好
        case highTension       = "持续紧张"      // 紧张度偏高
        case poorRecovery      = "恢复不足"      // 休息后回升不明显
        case signalDrop        = "信号中断"      // 中间有断点
    }

    func detectPatterns(for session: Session) -> [Pattern] {
        var patterns: [Pattern] = []
        let snapshots = session.segments.flatMap { $0.snapshots }.sorted { $0.timestamp < $1.timestamp }
        guard !snapshots.isEmpty else { return [] }

        let staminaValues = snapshots.map(\.stamina)
        let tensionValues = snapshots.map(\.tension)
        let count = staminaValues.count

        // 早期下降：前 1/3 时间内续航下降超过 20 点
        let earlyCount = max(count / 3, 1)
        let earlyAvg = staminaValues.prefix(earlyCount).reduce(0, +) / Double(earlyCount)
        let overallAvg = staminaValues.reduce(0, +) / Double(count)
        if earlyAvg - overallAvg > 20 {
            patterns.append(Pattern(
                type: .earlyDecline,
                description: "前 1/3 时间续航从 \(Int(earlyAvg)) 降至 \(Int(overallAvg))",
                data: ["earlyAvg": earlyAvg, "overallAvg": overallAvg]
            ))
        }

        // 稳定表现：标准差小于 10
        let avg = staminaValues.reduce(0, +) / Double(count)
        let variance = staminaValues.map { pow($0 - avg, 2) }.reduce(0, +) / Double(count)
        let stdDev = sqrt(variance)
        if stdDev < 10 && avg > 50 {
            patterns.append(Pattern(
                type: .stablePerformance,
                description: "续航波动较小（标准差 \(Int(stdDev))）",
                data: ["stdDev": stdDev, "avg": avg]
            ))
        }

        // 后程发力：后半段平均续航高于前半段 10 点以上
        let mid = count / 2
        let firstHalf = staminaValues.prefix(mid).reduce(0, +) / Double(max(mid, 1))
        let secondHalf = staminaValues.suffix(from: mid).reduce(0, +) / Double(max(count - mid, 1))
        if secondHalf - firstHalf > 10 {
            patterns.append(Pattern(
                type: .latePeak,
                description: "后半段续航（\(Int(secondHalf))）高于前半段（\(Int(firstHalf))）",
                data: ["firstHalf": firstHalf, "secondHalf": secondHalf]
            ))
        }

        // 持续紧张：40% 以上时间紧张度 > 0.5
        let highTensionRatio = tensionValues.filter { $0 > 0.5 }.count
        if Double(highTensionRatio) / Double(count) > 0.4 {
            patterns.append(Pattern(
                type: .highTension,
                description: "\(Int(Double(highTensionRatio) / Double(count) * 100))% 的时间紧张度偏高",
                data: ["ratio": Double(highTensionRatio) / Double(count)]
            ))
        }

        return patterns
    }

    func detectDailyPatterns(sessions: [Session]) -> [Pattern] {
        guard sessions.count >= 2 else { return [] }

        var patterns: [Pattern] = []
        let avgStaminas = sessions.compactMap(\.avgStamina).compactMap { $0 }

        // 时段分析
        var timeSlotData: [String: [Double]] = [:]
        for s in sessions {
            guard let avg = s.avgStamina else { continue }
            let hour = Calendar.current.component(.hour, from: s.startedAt)
            let slot: String
            switch hour {
            case 6..<12: slot = "上午"
            case 12..<14: slot = "午间"
            case 14..<18: slot = "下午"
            case 18..<22: slot = "晚间"
            default: slot = "其他"
            }
            timeSlotData[slot, default: []].append(avg)
        }

        // 找最佳时段
        let slotAvgs = timeSlotData.mapValues { $0.reduce(0, +) / Double($0.count) }
        if let best = slotAvgs.max(by: { $0.value < $1.value }) {
            patterns.append(Pattern(
                type: .stablePerformance,
                description: "\(best.key)平均续航 \(Int(best.value))，是最佳时段",
                data: ["timeSlot": best.key, "avg": best.value]
            ))
        }

        // 趋势分析
        if avgStaminas.count >= 2 {
            let firstHalf = avgStaminas.prefix(avgStaminas.count / 2).reduce(0, +) / Double(max(avgStaminas.count / 2, 1))
            let secondHalf = avgStaminas.suffix(from: avgStaminas.count / 2).reduce(0, +) / Double(max(avgStaminas.count / 2, 1))
            if secondHalf - firstHalf < -10 {
                patterns.append(Pattern(
                    type: .earlyDecline,
                    description: "后续航持续下降",
                    data: ["delta": secondHalf - firstHalf]
                ))
            }
        }

        return patterns
    }

    // MARK: - Stats Extraction

    struct SessionStats {
        let durationMinutes: Int
        let avgStamina: Double
        let minStamina: Double
        let staminaDelta: Double
        let avgTension: Double
        let avgFatigue: Double
        let avgConsistency: Double
        let segmentCount: Int
        let timeOfDay: String
    }

    func extractStats(_ session: Session) -> SessionStats {
        let snapshots = session.segments.flatMap { $0.snapshots }
        let staminaValues = snapshots.map(\.stamina)

        let durationMin = Int(session.duration / 60)
        let avg = staminaValues.isEmpty ? 0 : staminaValues.reduce(0, +) / Double(staminaValues.count)
        let min = staminaValues.min() ?? 0
        let delta = (staminaValues.last ?? 0) - (staminaValues.first ?? 0)
        let tension = snapshots.isEmpty ? 0 : snapshots.map(\.tension).reduce(0, +) / Double(snapshots.count)
        let fatigue = snapshots.isEmpty ? 0 : snapshots.map(\.fatigue).reduce(0, +) / Double(snapshots.count)
        let consistency = snapshots.isEmpty ? 0 : snapshots.map(\.consistency).reduce(0, +) / Double(snapshots.count)

        let hour = Calendar.current.component(.hour, from: session.startedAt)
        let timeOfDay: String
        switch hour {
        case 6..<12: timeOfDay = "上午"
        case 12..<14: timeOfDay = "午间"
        case 14..<18: timeOfDay = "下午"
        case 18..<22: timeOfDay = "晚间"
        default: timeOfDay = "其他"
        }

        return SessionStats(
            durationMinutes: durationMin,
            avgStamina: avg,
            minStamina: min,
            staminaDelta: delta,
            avgTension: tension,
            avgFatigue: fatigue,
            avgConsistency: consistency,
            segmentCount: session.segments.count,
            timeOfDay: timeOfDay
        )
    }

    struct DailyStats {
        let sessionCount: Int
        let totalMinutes: Int
        let avgStamina: Double
        let bestStamina: Double
        let worstStamina: Double
        let avgTension: Double
        let avgFatigue: Double
    }

    func extractDailyStats(_ sessions: [Session]) -> DailyStats {
        let totalMin = Int(sessions.reduce(0) { $0 + $1.duration } / 60)
        let staminas = sessions.compactMap(\.avgStamina).compactMap { $0 }
        let allSnapshots = sessions.flatMap { $0.segments.flatMap { $0.snapshots } }

        return DailyStats(
            sessionCount: sessions.count,
            totalMinutes: totalMin,
            avgStamina: staminas.isEmpty ? 0 : staminas.reduce(0, +) / Double(staminas.count),
            bestStamina: staminas.max() ?? 0,
            worstStamina: staminas.min() ?? 0,
            avgTension: allSnapshots.isEmpty ? 0 : allSnapshots.map(\.tension).reduce(0, +) / Double(allSnapshots.count),
            avgFatigue: allSnapshots.isEmpty ? 0 : allSnapshots.map(\.fatigue).reduce(0, +) / Double(allSnapshots.count)
        )
    }

    // MARK: - Prompt Builder

    private func buildInsightPrompt(_ s: SessionStats, patterns: [Pattern]) -> String {
        let patternDesc = patterns.isEmpty ? "无明显异常模式" : patterns.map { $0.description }.joined(separator: "；")

        return """
        请用 1-2 句话客观描述以下身体数据，不要给建议，只呈现事实：

        时段：\(s.timeOfDay)
        时长：\(s.durationMinutes) 分钟
        续航：平均 \(Int(s.avgStamina))，最低 \(Int(s.minStamina))，变化 \(s.staminaDelta > 0 ? "+" : "")\(Int(s.staminaDelta))
        肌肉紧张度：\(Int(s.avgTension * 100))%
        疲劳度：\(Int(s.avgFatigue * 100))%
        信号稳定性：\(Int(s.avgConsistency * 100))%
        模式：\(patternDesc)

        要求：
        - 只描述"身体显示了什么"，不写"建议"
        - 不用"应该"、"可以"、"试试"等指导性词语
        - 用客观陈述语气，像体检报告而非医生建议
        - 最多 2 句话
        """
    }

    private func buildDailyPrompt(_ s: DailyStats, patterns: [Pattern]) -> String {
        let patternDesc = patterns.isEmpty ? "无明显模式" : patterns.map { $0.description }.joined(separator: "；")

        return """
        请用 1-2 句话总结今天的身体状态数据，只呈现事实，不给建议：

        今日 \(s.sessionCount) 段专注，共 \(s.totalMinutes) 分钟
        续航：平均 \(Int(s.avgStamina))，最佳 \(Int(s.bestStamina))，最低 \(Int(s.worstStamina))
        肌肉紧张度：\(Int(s.avgTension * 100))%
        疲劳度：\(Int(s.avgFatigue * 100))%
        模式：\(patternDesc)

        要求：
        - 只描述"数据显示了什么"
        - 不给任何建议
        - 最多 2 句话
        """
    }

    // MARK: - Foundation Models

    private func tryFoundationModels(prompt: String) async -> String? {
        #if canImport(FoundationModels)
        let model = SystemLanguageModel.default
        guard model.isAvailable else { return nil }

        let instructions = """
        你是 FocuX 的体能状态解读引擎。

        定位：你是"镜子"而非"教练"——只呈现身体状态，不告诉用户该做什么。

        风格：
        - 客观、描述性、数据驱动
        - 不使用"建议"、"应该"、"可以"、"试试"等指导性词语
        - 使用"数据显示"、"身体信号"、"从数据看"等描述性开头
        - 像体检报告的解读，而非医生的建议

        禁用词汇：
        - "建议"、"试试"、"应该"、"可以"、"最好"、"不妨"
        - "保持"、"坚持"、"注意"、"记得"
        - "希望你"、"祝你"
        """

        do {
            let session = LanguageModelSession { instructions }
            let response = try await session.respond(to: prompt)
            let text = response.content.trimmingCharacters(in: .whitespacesAndNewlines)
            return text.isEmpty ? nil : text
        } catch {
            print("[BodyInsight] Foundation Models 错误: \(error)")
            return nil
        }
        #else
        return nil
        #endif
    }

    // MARK: - Fallback

    private func fallbackInsight(_ s: SessionStats, patterns: [Pattern]) -> String {
        var parts: [String] = []

        // 基于续航的开场
        if s.avgStamina >= 70 {
            parts.append("\(s.timeOfDay)这段 \(s.durationMinutes) 分钟续航平均 \(Int(s.avgStamina))，肌肉信号稳定。")
        } else if s.avgStamina >= 50 {
            parts.append("\(s.timeOfDay)这段 \(s.durationMinutes) 分钟续航平均 \(Int(s.avgStamina))，有波动。")
        } else {
            parts.append("\(s.timeOfDay)这段 \(s.durationMinutes) 分钟续航平均 \(Int(s.avgStamina))，身体信号显示疲劳。")
        }

        // 模式描述（最多一个）
        if let pattern = patterns.first {
            parts.append(pattern.description + "。")
        }

        // 维度补充（只描述，不给建议）
        if s.avgTension > 0.5 {
            parts.append("紧张度 \(Int(s.avgTension * 100))% 偏高。")
        } else if s.avgFatigue > 0.6 {
            parts.append("疲劳度达到 \(Int(s.avgFatigue * 100))%。")
        } else if s.staminaDelta < -15 {
            parts.append("续航下降了 \(Int(abs(s.staminaDelta))) 点。")
        }

        return parts.isEmpty ? "数据显示续航 \(Int(s.avgStamina))。" : parts.joined(separator: " ")
    }

    private func fallbackDailySummary(_ s: DailyStats, patterns: [Pattern]) -> String {
        var parts: [String] = []

        parts.append("今日 \(s.sessionCount) 段专注共 \(s.totalMinutes) 分钟，平均续航 \(Int(s.avgStamina))。")

        if let pattern = patterns.first {
            parts.append(pattern.description + "。")
        }

        if s.avgTension > 0.5 {
            parts.append("全天紧张度 \(Int(s.avgTension * 100))%。")
        } else if s.avgFatigue > 0.6 {
            parts.append("全天疲劳度 \(Int(s.avgFatigue * 100))%。")
        }

        return parts.joined(separator: "")
    }
}
