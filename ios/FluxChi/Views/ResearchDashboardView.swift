import SwiftUI
import SwiftData
import Charts

/**
 ResearchDashboardView - 研究/论文级数据仪表盘

 双层设计：
 1. 数据层 - 原始 EMG 数据、统计特征、时间序列
 2. 洞察层 - 个体化基线、模式识别、可执行建议
 */

struct ResearchDashboardView: View {
    @Environment(\.modelContext) private var modelContext
    @Query private var sessions: [Session]
    @Query private var feedbacks: [UserFeedback]

    @State private var selectedTab: DashboardTab = .data
    @State private var selectedSession: Session?
    @State private var showExportOptions = false

    // MARK: - Computed Properties

    private var recentSessions: [Session] {
        let calendar = Calendar.current
        let sevenDaysAgo = calendar.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        return sessions.filter { $0.startedAt >= sevenDaysAgo }
            .sorted { $0.startedAt > $1.startedAt }
    }

    private var aggregateStats: AggregateStats {
        computeAggregateStats(from: recentSessions)
    }

    var body: some View {
        VStack(spacing: 0) {
            // Tab 切换
            tabBar

            ScrollView {
                VStack(spacing: Flux.Spacing.section) {
                    switch selectedTab {
                    case .data:
                        dataLayerContent
                    case .insights:
                        insightsLayerContent
                    case .research:
                        researchLayerContent
                    }
                }
                .padding()
            }
        }
        .background(Color(.systemGroupedBackground))
        .sheet(item: $selectedSession) { session in
            SessionResearchSheet(session: session)
        }
        .sheet(isPresented: $showExportOptions) {
            ExportOptionsSheet()
        }
    }

    // MARK: - Tab Bar

    private var tabBar: some View {
        HStack(spacing: 0) {
            ForEach(DashboardTab.allCases) { tab in
                Button {
                    selectedTab = tab
                } label: {
                    VStack(spacing: 4) {
                        Image(systemName: tab.icon)
                            .font(.system(size: 20))
                        Text(tab.displayName)
                            .font(.caption2)
                    }
                    .frame(maxWidth: .infinity)
                    .foregroundStyle(selectedTab == tab ? .cyan : .secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.top, 8)
        .padding(.bottom, 4)
        .background(Color(.systemBackground))
    }

    // MARK: - Data Layer (原始数据)

    private var dataLayerContent: some View {
        VStack(alignment: .leading, spacing: 16) {
            // 统计卡片
            HStack(spacing: 12) {
                StatCard(title: "会话数", value: "\(recentSessions.count)", icon: "chart.bar", color: .blue)
                StatCard(title: "总时长", value: formatDuration(aggregateStats.totalDuration), icon: "clock", color: .green)
                StatCard(title: "平均续航", value: "\(Int(aggregateStats.avgStamina))", icon: "bolt.fill", color: .yellow)
                StatCard(title: "平均疲劳", value: "\(Int(aggregateStats.avgFatigue * 100))%", icon: "battery.25", color: .orange)
            }

            // 时间序列图表
            timeSeriesChart

            // 维度分布
            dimensionDistribution

            // 会话列表
            sessionList
        }
    }

    // MARK: - Insights Layer (智能洞察)

    private var insightsLayerContent: some View {
        VStack(alignment: .leading, spacing: 16) {
            // 个体化基线
            baselineCard

            // 趋势对比
            trendsCard

            // 模式识别
            patternRecognitionCard

            // 可执行建议
            actionableInsightsCard

            // 风险预警
            if !aggregateStats.risks.isEmpty {
                riskAlertCard
            }
        }
    }

    // MARK: - Research Layer (论文级数据)

    private var researchLayerContent: some View {
        VStack(alignment: .leading, spacing: 16) {
            // EMG 特征统计
            emgFeaturesCard

            // 相关性分析
            correlationCard

            // 导出选项
            exportCard
        }
    }

    // MARK: - Time Series Chart

    private var timeSeriesChart: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("续航趋势")
                .font(.headline)

            Chart {
                ForEach(Array(recentSessions.enumerated()), id: \.offset) { idx, session in
                    if let avg = session.avgStamina {
                        LineMark(
                            x: .value("时间", session.startedAt),
                            y: .value("续航", avg)
                        )
                        .foregroundStyle(.cyan)
                        .interpolationMethod(.catmullRom)

                        PointMark(
                            x: .value("时间", session.startedAt),
                            y: .value("续航", avg)
                        )
                        .foregroundStyle(.cyan)
                        .annotation(position: .top) {
                            Text("\(Int(avg))")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .frame(height: 180)
            .chartYAxis {
                AxisMarks(position: .leading)
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))
    }

    // MARK: - Dimension Distribution

    private var dimensionDistribution: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("维度分析")
                .font(.headline)

            HStack(spacing: 16) {
                // 一致性
                DimensionBar(
                    title: "一致性",
                    value: aggregateStats.avgConsistency,
                    color: .blue
                )

                // 紧张度
                DimensionBar(
                    title: "紧张度",
                    value: aggregateStats.avgTension,
                    color: .orange
                )

                // 疲劳度
                DimensionBar(
                    title: "疲劳度",
                    value: aggregateStats.avgFatigue,
                    color: .red
                )
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))
    }

    // MARK: - Session List

    private var sessionList: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("会话记录")
                    .font(.headline)
                Spacer()
                Button("导出全部") {
                    showExportOptions = true
                }
                .font(.caption)
            }

            ForEach(recentSessions.prefix(5)) { session in
                SessionRow(session: session)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        selectedSession = session
                    }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))
    }

    // MARK: - Baseline Card

    private var baselineCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "person.text.rectangle")
                    .foregroundStyle(.purple)
                Text("个体化基线")
                    .font(.headline)
            }

            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                GridRow {
                    Text("平均续航")
                        .foregroundStyle(.secondary)
                    Text("\(Int(aggregateStats.avgStamina))")
                        .bold()
                }

                GridRow {
                    Text("最佳时段")
                        .foregroundStyle(.secondary)
                    Text(aggregateStats.bestTimeSlot)
                        .bold()
                }

                GridRow {
                    Text("数据记录")
                        .foregroundStyle(.secondary)
                    Text("\(recentSessions.count) 天")
                        .bold()
                }
            }
            .font(.subheadline)
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))
    }

    // MARK: - Trends Card

    private var trendsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .foregroundStyle(.blue)
                Text("趋势对比")
                    .font(.headline)
            }

            HStack(spacing: 20) {
                TrendItem(label: "比昨天", value: aggregateStats.vsYesterday, color: .blue)
                TrendItem(label: "比上周", value: aggregateStats.vsLastWeek, color: .purple)
                TrendItem(label: "变化率", value: aggregateStats.changeRate, color: .green)
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))
    }

    // MARK: - Pattern Recognition Card

    private var patternRecognitionCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundStyle(.pink)
                Text("模式识别")
                    .font(.headline)
            }

            VStack(alignment: .leading, spacing: 8) {
                ForEach(aggregateStats.patterns, id: \.type) { pattern in
                    HStack {
                        Circle()
                            .fill(pattern.severity.color)
                            .frame(width: 8, height: 8)
                        Text(pattern.description)
                            .font(.subheadline)
                    }
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))
    }

    // MARK: - Actionable Insights Card

    private var actionableInsightsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "lightbulb.fill")
                    .foregroundStyle(.yellow)
                Text("可执行建议")
                    .font(.headline)
            }

            Text("基于你的数据，建议：")
                .font(.caption)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 8) {
                ForEach(aggregateStats.actions, id: \.title) { action in
                    HStack(alignment: .top, spacing: 12) {
                        Text(action.emoji)
                            .font(.title3)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(action.title)
                                .font(.subheadline.bold())
                            Text(action.description)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))
    }

    // MARK: - Risk Alert Card

    private var riskAlertCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                Text("风险预警")
                    .font(.headline)
            }

            ForEach(aggregateStats.risks, id: \.type) { risk in
                HStack {
                    Image(systemName: "arrow.right.circle.fill")
                        .font(.caption)
                        .foregroundStyle(risk.severity.color)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(risk.title)
                            .font(.subheadline.bold())
                        Text(risk.description)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .padding()
        .background(Color.red.opacity(0.1), in: .rect(cornerRadius: 16))
    }

    // MARK: - EMG Features Card

    private var emgFeaturesCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "waveform.path")
                    .foregroundStyle(.cyan)
                Text("EMG 特征统计")
                    .font(.headline)
            }

            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                GridRow {
                    Text("采样点数")
                        .foregroundStyle(.secondary)
                    Text("\(aggregateStats.totalSnapshots)")
                        .bold()
                }

                GridRow {
                    Text("有效通道")
                        .foregroundStyle(.secondary)
                    Text("\(aggregateStats.activeChannels)")
                        .bold()
                }

                GridRow {
                    Text("平均 RMS")
                        .foregroundStyle(.secondary)
                    Text(String(format: "%.2f", aggregateStats.avgRMS))
                        .bold()
                }

                GridRow {
                    Text("峰值 RMS")
                        .foregroundStyle(.secondary)
                    Text(String(format: "%.2f", aggregateStats.peakRMS))
                        .bold()
                }
            }
            .font(.subheadline)
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))
    }

    // MARK: - Correlation Card

    private var correlationCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "chart.xyaxis.line")
                    .foregroundStyle(.indigo)
                Text("相关性分析")
                    .font(.headline)
            }

            VStack(alignment: .leading, spacing: 8) {
                ForEach(aggregateStats.correlations, id: \.pair) { corr in
                    HStack {
                        Text(corr.pair)
                            .font(.caption)
                            .frame(width: 80, alignment: .leading)
                        ProgressView(value: abs(corr.value))
                            .tint(corr.value > 0 ? .green : .red)
                        Text(String(format: "%.2f", corr.value))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(width: 40, alignment: .trailing)
                    }
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))
    }

    // MARK: - Export Card

    private var exportCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "square.and.arrow.up")
                    .foregroundStyle(.gray)
                Text("数据导出")
                    .font(.headline)
            }

            VStack(spacing: 8) {
                ExportButton(title: "导出 JSON", icon: "doc.text", format: .json)
                ExportButton(title: "导出 CSV", icon: "tablecells", format: .csv)
                ExportButton(title: "同步到 Apple Health", icon: "heart.fill", format: .healthkit)
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))
    }

    // MARK: - Helper Views

    struct StatCard: View {
        let title: String
        let value: String
        let icon: String
        let color: Color

        var body: some View {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .foregroundStyle(color)
                Text(value)
                    .font(.title2.bold())
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 12))
        }
    }

    struct DimensionBar: View {
        let title: String
        let value: Double
        let color: Color

        var body: some View {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color(.systemGray5))
                        .frame(height: 8)

                    RoundedRectangle(cornerRadius: 4)
                        .fill(color.gradient)
                        .frame(width: max(0, CGFloat(value) * 100), height: 8)
                }

                Text("\(Int(value * 100))%")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    struct SessionRow: View {
        let session: Session

        var body: some View {
            HStack(spacing: 12) {
                // 状态指示器
                Circle()
                    .fill(sessionColor)
                    .frame(width: 12, height: 12)

                VStack(alignment: .leading, spacing: 2) {
                    Text(session.title.isEmpty ? "专注会话" : session.title)
                        .font(.subheadline)
                    Text(session.startedAt, style: .date)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 2) {
                    Text("\(session.avgStamina.map { Int($0) } ?? 0)")
                        .font(.subheadline.bold())
                    Text("\(Int(session.duration / 60)) 分钟")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 8)
        }

        private var sessionColor: Color {
            guard let avg = session.avgStamina else { return .gray }
            if avg >= 60 { return .green }
            if avg >= 30 { return .orange }
            return .red
        }

        private var startDate: Date { session.startedAt }
        private var endDate: Date { session.endedAt ?? Date() }
    }

    struct TrendItem: View {
        let label: String
        let value: String
        let color: Color

        var body: some View {
            VStack(spacing: 4) {
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(value)
                    .font(.subheadline.bold())
                    .foregroundStyle(color)
            }
        }
    }

    struct ExportButton: View {
        let title: String
        let icon: String
        let format: ExportFormat

        var body: some View {
            Button {
                // TODO: 实现导出
            } label: {
                HStack {
                    Image(systemName: icon)
                        .frame(width: 24)
                    Text(title)
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                .padding()
                .background(Color(.tertiarySystemBackground), in: .rect(cornerRadius: 8))
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - Data Models

    enum DashboardTab: String, CaseIterable, Identifiable {
        case data, insights, research

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .data: return "数据"
            case .insights: return "洞察"
            case .research: return "研究"
            }
        }

        var icon: String {
            switch self {
            case .data: return "chart.bar"
            case .insights: return "lightbulb"
            case .research: return "microscope"
            }
        }
    }

    struct AggregateStats {
        let totalDuration: TimeInterval
        let avgStamina: Double
        let avgConsistency: Double
        let avgTension: Double
        let avgFatigue: Double
        let bestTimeSlot: String
        let vsYesterday: String
        let vsLastWeek: String
        let changeRate: String
        let patterns: [Pattern]
        let actions: [ActionableInsight]
        let risks: [RiskAlert]
        let totalSnapshots: Int
        let activeChannels: Int
        let avgRMS: Double
        let peakRMS: Double
        let correlations: [Correlation]
    }

    struct Pattern {
        let type: String
        let description: String
        let severity: Severity
    }

    struct Severity {
        let color: Color
    }

    struct ActionableInsight {
        let title: String
        let description: String
        let emoji: String
    }

    struct RiskAlert {
        let type: String
        let title: String
        let description: String
        let severity: Severity
    }

    struct Correlation {
        let pair: String
        let value: Double
    }

    enum ExportFormat {
        case json, csv, healthkit
    }

    // MARK: - Computation Methods

    private func computeAggregateStats(from sessions: [Session]) -> AggregateStats {
        guard !sessions.isEmpty else {
            return emptyStats()
        }

        let calendar = Calendar.current
        let now = Date()

        // 基础统计
        let totalDuration = sessions.reduce(0) { $0 + $1.duration }
        let avgStamina = sessions.compactMap(\.avgStamina).reduce(0, +) / Double(sessions.count)

        // 维度统计
        let allSnapshots = sessions.flatMap { $0.segments.flatMap { $0.snapshots } }
        let avgConsistency = allSnapshots.map(\.consistency).reduce(0, +) / Double(max(1, allSnapshots.count))
        let avgTension = allSnapshots.map(\.tension).reduce(0, +) / Double(max(1, allSnapshots.count))
        let avgFatigue = allSnapshots.map(\.fatigue).reduce(0, +) / Double(max(1, allSnapshots.count))

        // EMG 统计
        let totalSnapshots = allSnapshots.count
        let allRMS = allSnapshots.flatMap { $0.rms }
        let avgRMS = allRMS.reduce(0, +) / Double(max(1, allRMS.count))
        let peakRMS = allRMS.max() ?? 0
        let activeChannels = detectActiveChannels(from: allSnapshots)

        // 最佳时段
        let bestTimeSlot = findBestTimeSlot(from: sessions)

        // 趋势
        let yesterdayStart = calendar.date(byAdding: .day, value: -1, to: calendar.startOfDay(for: now)) ?? now
        let yesterdayEnd = calendar.startOfDay(for: now)
        let yesterdaySessions = sessions.filter { $0.startedAt >= yesterdayStart && $0.startedAt < yesterdayEnd }
        let yesterdayAvg = yesterdaySessions.compactMap(\.avgStamina).reduce(0, +) / Double(max(1, yesterdaySessions.count))

        let vsYesterday = avgStamina - yesterdayAvg

        // 模式识别
        let patterns = detectPatterns(from: sessions, snapshots: allSnapshots)

        // 风险预警
        let risks = detectRisks(avgFatigue: avgFatigue, avgTension: avgTension, sessions: sessions)

        // 可执行建议
        let actions = generateActions(fatigue: avgFatigue, tension: avgTension, bestTimeSlot: bestTimeSlot)

        // 相关性分析
        let correlations = computeCorrelations(from: allSnapshots)

        return AggregateStats(
            totalDuration: totalDuration,
            avgStamina: avgStamina,
            avgConsistency: avgConsistency,
            avgTension: avgTension,
            avgFatigue: avgFatigue,
            bestTimeSlot: bestTimeSlot,
            vsYesterday: formatChange(vsYesterday),
            vsLastWeek: "+0",  // TODO: 实现
            changeRate: "0%",   // TODO: 实现
            patterns: patterns,
            actions: actions,
            risks: risks,
            totalSnapshots: totalSnapshots,
            activeChannels: activeChannels,
            avgRMS: avgRMS,
            peakRMS: peakRMS,
            correlations: correlations
        )
    }

    private func emptyStats() -> AggregateStats {
        AggregateStats(
            totalDuration: 0,
            avgStamina: 0,
            avgConsistency: 0,
            avgTension: 0,
            avgFatigue: 0,
            bestTimeSlot: "暂无",
            vsYesterday: "—",
            vsLastWeek: "—",
            changeRate: "—",
            patterns: [],
            actions: [],
            risks: [],
            totalSnapshots: 0,
            activeChannels: 0,
            avgRMS: 0,
            peakRMS: 0,
            correlations: []
        )
    }

    private func detectActiveChannels(from snapshots: [FluxSnapshot]) -> Int {
        let sampleCount = min(snapshots.count, 100)
        let sampled = snapshots.suffix(sampleCount)
        let ch7HasData = sampled.contains { abs($0.rms6) > 0.01 }
        let ch8HasData = sampled.contains { abs($0.rms7) > 0.01 }
        return (ch7HasData || ch8HasData) ? 8 : 6
    }

    private func findBestTimeSlot(from sessions: [Session]) -> String {
        let calendar = Calendar.current
        var slotScores: [String: [Double]] = [:]

        for s in sessions {
            guard let stamina = s.avgStamina else { continue }
            let hour = calendar.component(.hour, from: s.startedAt)
            let slot: String
            switch hour {
            case 6..<12: slot = "上午"
            case 12..<14: slot = "午间"
            case 14..<18: slot = "下午"
            case 18..<22: slot = "晚间"
            default: slot = "其他"
            }
            slotScores[slot, default: []].append(stamina)
        }

        return slotScores
            .mapValues { $0.reduce(0, +) / Double($0.count) }
            .max { $0.value < $1.value }?
            .key ?? "暂无"
    }

    private func detectPatterns(from sessions: [Session], snapshots: [FluxSnapshot]) -> [Pattern] {
        var patterns: [Pattern] = []

        // 早期下降模式
        let staminaValues = snapshots.map(\.stamina)
        if staminaValues.count > 10 {
            let firstThird = staminaValues.prefix(staminaValues.count / 3).reduce(0, +) / Double(max(1, staminaValues.count / 3))
            let overall = staminaValues.reduce(0, +) / Double(staminaValues.count)
            if firstThird - overall > 15 {
                patterns.append(Pattern(
                    type: "early_decline",
                    description: "前半段续航下降明显，可能热身不足",
                    severity: Severity(color: .orange)
                ))
            }
        }

        // 高紧张模式
        let highTensionRatio = snapshots.filter { $0.tension > 0.5 }.count
        if Double(highTensionRatio) / Double(snapshots.count) > 0.3 {
            patterns.append(Pattern(
                type: "high_tension",
                description: "\(Int(Double(highTensionRatio) / Double(snapshots.count) * 100))% 时间紧张度偏高",
                severity: Severity(color: .yellow)
            ))
        }

        return patterns
    }

    private func detectRisks(avgFatigue: Double, avgTension: Double, sessions: [Session]) -> [RiskAlert] {
        var risks: [RiskAlert] = []

        if avgFatigue > 0.7 {
            risks.append(RiskAlert(
                type: "high_fatigue",
                title: "疲劳累积",
                description: "疲劳度 \(Int(avgFatigue * 100))% 已达警戒线，建议增加休息",
                severity: Severity(color: .red)
            ))
        }

        if avgTension > 0.6 {
            risks.append(RiskAlert(
                type: "high_tension",
                title: "肌肉紧张",
                description: "持续紧张可能增加 RSI 风险，注意放松",
                severity: Severity(color: .orange)
            ))
        }

        return risks
    }

    private func generateActions(fatigue: Double, tension: Double, bestTimeSlot: String) -> [ActionableInsight] {
        var actions: [ActionableInsight] = []

        if tension > 0.5 {
            actions.append(ActionableInsight(
                title: "每小时放松",
                description: "肩部画圈 5 次、深呼吸 3 次",
                emoji: "🧘"
            ))
        }

        if fatigue > 0.6 {
            actions.append(ActionableInsight(
                title: "主动休息",
                description: "25 分钟工作 + 5 分钟休息",
                emoji: "⏱️"
            ))
        }

        if bestTimeSlot != "暂无" {
            actions.append(ActionableInsight(
                title: "利用最佳时段",
                description: "重要任务放在 \(bestTimeSlot)",
                emoji: "🌟"
            ))
        }

        return actions.isEmpty ? [
            ActionableInsight(title: "保持节奏", description: "稳定的工作休息模式", emoji: "📊")
        ] : actions
    }

    private func computeCorrelations(from snapshots: [FluxSnapshot]) -> [Correlation] {
        guard snapshots.count > 10 else { return [] }

        var correlations: [Correlation] = []

        // 紧张度 vs 疲劳度相关性
        let tension = snapshots.map(\.tension)
        let fatigue = snapshots.map(\.fatigue)
        let corr = pearsonCorrelation(tension, fatigue)

        correlations.append(Correlation(pair: "紧张-疲劳", value: corr))

        // 一致性 vs 续航相关性
        let consistency = snapshots.map(\.consistency)
        let stamina = snapshots.map(\.stamina)
        let corr2 = pearsonCorrelation(consistency, stamina)

        correlations.append(Correlation(pair: "一致性-续航", value: corr2))

        return correlations
    }

    private func pearsonCorrelation(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count, !x.isEmpty else { return 0 }

        let n = Double(x.count)
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumX2 = x.map { $0 * $0 }.reduce(0, +)
        let sumY2 = y.map { $0 * $0 }.reduce(0, +)

        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))

        return denominator == 0 ? 0 : numerator / denominator
    }

    private func formatChange(_ value: Double) -> String {
        if value > 0 {
            return "+\(Int(value))"
        } else if value < 0 {
            return "\(Int(value))"
        } else {
            return "持平"
        }
    }

    private func formatDuration(_ interval: TimeInterval) -> String {
        let minutes = Int(interval / 60)
        if minutes < 60 {
            return "\(minutes) 分钟"
        }
        let hours = minutes / 60
        let mins = minutes % 60
        return "\(hours) 小时 \(mins) 分钟"
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MM/dd HH:mm"
        return formatter.string(from: date)
    }
}

// MARK: - Session Research Sheet

struct SessionResearchSheet: View {
    let session: Session
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // 基本信息
                    infoSection

                    // EMG 数据
                    emgDataSection

                    // 导出
                    exportSection
                }
                .padding()
            }
            .navigationTitle("会话详情")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("关闭") { dismiss() }
                }
            }
        }
    }

    private var infoSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("基本信息")
                .font(.headline)

            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                GridRow { Text("开始时间"); Text(session.startedAt, style: .date) }
                GridRow { Text("结束时间"); Text((session.endedAt ?? Date()), style: .date) }
                GridRow { Text("总时长"); Text("\(Int(session.duration / 60)) 分钟") }
                GridRow { Text("平均续航"); Text("\(session.avgStamina.map { Int($0) } ?? 0)") }
            }
            .font(.subheadline)
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 12))
    }

    private var emgDataSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("EMG 数据")
                .font(.headline)

            Text("采样点: \(allSnapshots.count)")
                .font(.caption)
                .foregroundStyle(.secondary)

            // 数据概览
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                GridRow { Text("平均一致性"); Text("\(Int(avgConsistency * 100))%") }
                GridRow { Text("平均紧张度"); Text("\(Int(avgTension * 100))%") }
                GridRow { Text("平均疲劳度"); Text("\(Int(avgFatigue * 100))%") }
            }
            .font(.subheadline)
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 12))
    }

    private var exportSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("导出数据")
                .font(.headline)

            HStack(spacing: 12) {
                Button("JSON") { /* TODO */ }
                Button("CSV") { /* TODO */ }
                Button("Apple Health") { /* TODO */ }
            }
            .buttonStyle(.bordered)
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 12))
    }

    private var allSnapshots: [FluxSnapshot] {
        session.segments.flatMap { $0.snapshots }
    }

    private var avgConsistency: Double {
        let snaps = allSnapshots
        return snaps.isEmpty ? 0 : snaps.map(\.consistency).reduce(0, +) / Double(snaps.count)
    }

    private var avgTension: Double {
        let snaps = allSnapshots
        return snaps.isEmpty ? 0 : snaps.map(\.tension).reduce(0, +) / Double(snaps.count)
    }

    private var avgFatigue: Double {
        let snaps = allSnapshots
        return snaps.isEmpty ? 0 : snaps.map(\.fatigue).reduce(0, +) / Double(snaps.count)
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MM/dd HH:mm:ss"
        return formatter.string(from: date)
    }

    private func formatDuration(_ interval: TimeInterval) -> String {
        let minutes = Int(interval / 60)
        return "\(minutes) 分钟"
    }
}

// MARK: - Export Options Sheet

struct ExportOptionsSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var selectedFormat: ExportFormatOption = .json
    @State private var isExporting = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                Text("选择导出格式")
                    .font(.headline)

                Picker("格式", selection: $selectedFormat) {
                    ForEach(ExportFormatOption.allCases) { format in
                        Text(format.displayName).tag(format)
                    }
                }
                .pickerStyle(.segmented)

                VStack(alignment: .leading, spacing: 8) {
                    Text(formatDescription)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if isExporting {
                    ProgressView("导出中...")
                }

                Button("导出") {
                    performExport()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isExporting)
            }
            .padding()
            .navigationTitle("导出数据")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("取消") { dismiss() }
                }
            }
        }
    }

    var formatDescription: String {
        switch selectedFormat {
        case .json:
            return "完整数据，包含所有 EMG 原始采样点和特征值"
        case .csv:
            return "表格格式，适合 Excel 或 Python 分析"
        case .healthkit:
            return "同步到 Apple Health，与健康数据一起查看"
        }
    }

    private func performExport() {
        isExporting = true
        Task {
            // TODO: 实际导出逻辑
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            await MainActor.run {
                isExporting = false
                dismiss()
            }
        }
    }
}

enum ExportFormatOption: String, CaseIterable, Identifiable {
    case json, csv, healthkit

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .json: return "JSON"
        case .csv: return "CSV"
        case .healthkit: return "Apple Health"
        }
    }
}

#Preview {
    ResearchDashboardView()
        .modelContainer(for: Session.self, inMemory: true)
}
