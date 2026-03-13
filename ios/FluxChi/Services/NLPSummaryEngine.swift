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

    // MARK: - Foundation Models (on-device LLM)

    private func tryFoundationModels(prompt: String) async -> String? {
        #if canImport(FoundationModels)
        guard SystemLanguageModel.default.isAvailable else {
            print("[NLP] Foundation Models not available on this device")
            return nil
        }
        do {
            let instructions = """
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
            let session = LanguageModelSession(instructions: instructions)
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
