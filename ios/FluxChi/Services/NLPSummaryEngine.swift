import Foundation
import SwiftUI

/// NLP Summary Engine — thin facade delegating to NLP/ submodules.
/// All public APIs and nested type names are preserved for caller compatibility.
@available(iOS 26.0, *)
final class NLPSummaryEngine {

    static let shared = NLPSummaryEngine()

    // MARK: - Type Aliases (preserve NLPSummaryEngine.XXX access)

    typealias Persona       = NLPPersona
    typealias AnomalyType   = NLPAnomalyType
    typealias Severity      = NLPSeverity
    typealias Anomaly       = NLPAnomaly
    typealias SessionStats  = NLPSessionStats
    typealias DailyStats    = NLPDailyStats
    typealias WeeklyTrend   = NLPWeeklyTrend
    typealias CoachContext  = NLPCoachContext

    // MARK: - Model Bridge (delegated)

    var isAvailable: Bool { NLPModelBridge.isAvailable }
    var diagnosticInfo: String { NLPModelBridge.diagnosticInfo }
    func prewarm() { NLPModelBridge.prewarm() }

    // MARK: - Public API

    /// Generate AI summary for a session (prefer on-device LLM, fallback to template)
    func generateSummary(for session: Session) async -> String {
        let stats = NLPStatsExtractor.extractSessionStats(session)
        let anomalies = NLPAnomalyDetector.detectAnomalies(for: session)
        let prompt = NLPPromptBuilder.buildSessionPrompt(stats, anomalies: anomalies)

        if let nlpResult = await NLPModelBridge.tryGenerate(prompt: prompt) {
            return nlpResult
        }
        return NLPFallbackGenerator.sessionSummary(stats, anomalies: anomalies)
    }

    /// Generate AI advice for a session (used by SessionSummarySheet)
    func generateAdvice(for session: Session) async -> String? {
        let stats = NLPStatsExtractor.extractSessionStats(session)
        let prompt = NLPPromptBuilder.buildAdvicePrompt(stats)
        return await NLPModelBridge.tryGenerate(prompt: prompt)
    }

    // MARK: - Daily Insight

    /// Generate daily insight across all today's sessions
    func generateDailyInsight(sessions: [Session], allRecentSessions: [Session] = []) async -> String {
        guard !sessions.isEmpty else { return "今天还没有专注记录，开始第一段吧。" }

        let stats = NLPStatsExtractor.extractDailyStats(sessions)
        let anomalies = NLPAnomalyDetector.detectDailyAnomalies(sessions: sessions)
        let weeklyTrend = allRecentSessions.count >= 2
            ? NLPStatsExtractor.generateWeeklyTrend(sessions: allRecentSessions)
            : nil
        let prompt = NLPPromptBuilder.buildDailyPrompt(stats, anomalies: anomalies, weeklyTrend: weeklyTrend)

        if let nlpResult = await NLPModelBridge.tryGenerate(prompt: prompt, persona: .dailyCoach) {
            return nlpResult
        }
        return NLPFallbackGenerator.dailyInsight(stats, anomalies: anomalies)
    }

    /// Coach insight when no sessions exist today
    func generateEmptyDayInsight(recentSessions: [Session] = []) -> String {
        NLPFallbackGenerator.emptyDayInsight(recentSessions: recentSessions)
    }

    // MARK: - Weekly Trend

    /// Analyze the past 7 days trend
    func generateWeeklyTrend(sessions: [Session]) -> WeeklyTrend {
        NLPStatsExtractor.generateWeeklyTrend(sessions: sessions)
    }

    // MARK: - Anomaly Detection

    /// Detect anomalies for a single session
    func detectAnomalies(for session: Session) -> [Anomaly] {
        NLPAnomalyDetector.detectAnomalies(for: session)
    }

    /// Detect daily aggregated anomalies
    func detectDailyAnomalies(sessions: [Session]) -> [Anomaly] {
        NLPAnomalyDetector.detectDailyAnomalies(sessions: sessions)
    }

    // MARK: - Stats Extraction

    func extractStats(_ session: Session) -> SessionStats {
        NLPStatsExtractor.extractSessionStats(session)
    }

    // MARK: - Coach Follow-Up

    /// Preset follow-up questions
    static let presetQuestions: [String] = [
        "为什么我下午续航总是下降？",
        "怎样延长高效时间？",
        "我的紧张度正常吗？"
    ]

    /// Smart follow-up: answer user question based on current context
    func askFollowUp(context: CoachContext, question: String) async -> String {
        let stats = context.todaySessions.isEmpty
            ? nil
            : NLPStatsExtractor.extractDailyStats(context.todaySessions)
        let prompt = NLPPromptBuilder.buildFollowUpPrompt(context: context, stats: stats, question: question)

        if let nlpResult = await NLPModelBridge.tryGenerate(prompt: prompt, persona: .dailyCoach) {
            return nlpResult
        }
        return NLPFallbackGenerator.followUp(question: question, stats: stats, anomalies: context.anomalies)
    }

    // MARK: - Fallback (kept public for external use)

    func fallbackSummary(_ s: SessionStats, anomalies: [Anomaly] = []) -> String {
        NLPFallbackGenerator.sessionSummary(s, anomalies: anomalies)
    }
}
