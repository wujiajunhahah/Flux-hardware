import SwiftUI
import Charts

/// FluxChi 的 AI 洞察呈现层。
///
/// 设计参考 Oura Ring / Whoop / Levels 的 insight 模式：
/// - **Hero 卡**：大标题 + 量化次行 + 趋势 chip + AI narrative + 3 个 dimension chip + 单一深入入口。
/// - **Detail Sheet**：策划好的分块阅读体验（今日走势 → 身体信号 → 模式 → 7 天对比），不是聊天框。
/// - **聊天降级**：保留 ask-coach 能力但藏在 detail 底部三级链接，避免聊天框成为主要入口。

// MARK: - InsightStats（从 sessions 推导出展示用字段）

struct InsightStats {
    let today: [Session]
    let recent: [Session]

    init(today: [Session], recent: [Session]) {
        self.today = today
        self.recent = recent
    }

    // MARK: - Headline / Sub

    /// 时间感知 + 状态感知的大标题。短、平静、不夸张。
    var headline: String {
        if today.isEmpty {
            let hour = Calendar.current.component(.hour, from: Date())
            if hour < 12  { return "新的一天" }
            if hour < 18  { return "今天还在等待" }
            return "今晚静下来"
        }
        let s = avgStamina
        if s >= 75 { return "今天身体在线" }
        if s >= 60 { return "状态在调整" }
        if s >= 40 { return "今天有起伏" }
        return "身体在说慢一点"
    }

    /// 量化次行：N 段 · M 分钟 · 续航 X
    var subLine: String {
        guard !today.isEmpty else { return "暂无今日数据" }
        let mins = Int(today.reduce(0) { $0 + $1.duration } / 60)
        return "\(today.count) 段 · \(mins) 分钟 · 续航 \(Int(avgStamina))"
    }

    /// vs 昨天的对比；差距 < 3 时不显示，避免噪音。
    var deltaText: String? {
        guard !today.isEmpty, let prev = previousDayAvg else { return nil }
        let diff = Int(avgStamina - prev)
        guard abs(diff) >= 3 else { return nil }
        return diff > 0 ? "+\(diff) 比昨天" : "\(diff) 比昨天"
    }

    var deltaIsPositive: Bool {
        guard let prev = previousDayAvg else { return true }
        return avgStamina >= prev
    }

    /// Hero 卡上的 3 个 dimension chip — 只显示有意义的（最佳时段 / 高紧张 / 高疲劳）。
    var chips: [InsightChipData] {
        guard !today.isEmpty else { return [] }
        var result: [InsightChipData] = []
        if let best = bestSlot {
            result.append(.init(label: "最佳", value: "\(best.0.rawValue) \(Int(best.1))", tint: Flux.Colors.success))
        }
        if avgTension > 0.4 {
            result.append(.init(label: "紧张", value: "\(Int(avgTension * 100))%", tint: Flux.Colors.warning))
        }
        if avgFatigue > 0.5 {
            result.append(.init(label: "疲劳", value: "\(Int(avgFatigue * 100))%", tint: Flux.Colors.accent))
        }
        return Array(result.prefix(3))
    }

    /// AI 文本不可用时的兜底（短，一行）。
    var fallbackText: String {
        if today.isEmpty {
            let hour = Calendar.current.component(.hour, from: Date())
            if hour < 12 { return "新的一天，等你的第一段专注。" }
            if hour < 18 { return "今天还没有数据，找个 25 分钟试试？" }
            return "今晚静下来，明天再开始。"
        }
        let mins = Int(today.reduce(0) { $0 + $1.duration } / 60)
        return "今天 \(today.count) 段共 \(mins) 分钟，平均续航 \(Int(avgStamina))。"
    }

    // MARK: - Aggregates

    var avgStamina: Double {
        let v = today.compactMap(\.avgStamina)
        guard !v.isEmpty else { return 0 }
        return v.reduce(0, +) / Double(v.count)
    }

    var avgTension: Double {
        let s = today.flatMap { $0.segments.flatMap { $0.snapshots } }
        guard !s.isEmpty else { return 0 }
        return s.map(\.tension).reduce(0, +) / Double(s.count)
    }

    var avgFatigue: Double {
        let s = today.flatMap { $0.segments.flatMap { $0.snapshots } }
        guard !s.isEmpty else { return 0 }
        return s.map(\.fatigue).reduce(0, +) / Double(s.count)
    }

    var avgConsistency: Double {
        let s = today.flatMap { $0.segments.flatMap { $0.snapshots } }
        guard !s.isEmpty else { return 0 }
        return s.map(\.consistency).reduce(0, +) / Double(s.count)
    }

    var previousDayAvg: Double? {
        let cal = Calendar.current
        guard let yesterday = cal.date(byAdding: .day, value: -1, to: Date()) else { return nil }
        let yStart = cal.startOfDay(for: yesterday)
        let yEnd = cal.startOfDay(for: Date())
        let yes = recent.filter { $0.startedAt >= yStart && $0.startedAt < yEnd }
        let v = yes.compactMap(\.avgStamina)
        guard !v.isEmpty else { return nil }
        return v.reduce(0, +) / Double(v.count)
    }

    var bestSlot: (Flux.TimeSlot, Double)? {
        var sums: [Flux.TimeSlot: (sum: Double, n: Int)] = [:]
        for s in today {
            guard let avg = s.avgStamina else { continue }
            let slot = Flux.TimeSlot.from(date: s.startedAt)
            let cur = sums[slot] ?? (0, 0)
            sums[slot] = (cur.sum + avg, cur.n + 1)
        }
        return sums.map { ($0.key, $0.value.sum / Double($0.value.n)) }
            .max(by: { $0.1 < $1.1 })
    }
}

struct InsightChipData {
    let label: String
    let value: String
    let tint: Color
}

// MARK: - Hero Card（Dashboard 上的入口）

struct DailyInsightHeroCard: View {
    let todaySessions: [Session]
    let recentSessions: [Session]

    @State private var insightText: String?
    @State private var isLoading = false
    @State private var showDetail = false

    private var stats: InsightStats {
        InsightStats(today: todaySessions, recent: recentSessions)
    }

    var body: some View {
        Button {
            showDetail = true
        } label: {
            VStack(alignment: .leading, spacing: 16) {
                headlineBlock
                if let text = insightText {
                    Text(text)
                        .font(.system(size: 14))
                        .foregroundStyle(.primary.opacity(0.82))
                        .lineSpacing(5)
                        .fixedSize(horizontal: false, vertical: true)
                        .multilineTextAlignment(.leading)
                } else if isLoading {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("正在生成今日洞察…")
                            .font(.system(size: 13))
                            .foregroundStyle(.tertiary)
                    }
                }
                if !stats.chips.isEmpty {
                    chipsRow
                }
                cta
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(20)
            .background(cardBackground)
        }
        .buttonStyle(.plain)
        .sheet(isPresented: $showDetail) {
            InsightDetailView(
                todaySessions: todaySessions,
                recentSessions: recentSessions,
                insightText: insightText
            )
        }
        .task(id: todaySessions.count) {
            await loadInsight()
        }
    }

    // MARK: - Subviews

    private var headlineBlock: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(stats.headline)
                .font(.system(size: 26, weight: .semibold, design: .rounded))
                .foregroundStyle(.primary)

            HStack(spacing: 8) {
                Text(stats.subLine)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(.secondary)
                if let delta = stats.deltaText {
                    trendChip(text: delta, up: stats.deltaIsPositive)
                }
            }
        }
    }

    private var chipsRow: some View {
        HStack(spacing: 8) {
            ForEach(stats.chips, id: \.label) { chip in
                InsightChip(label: chip.label, value: chip.value, tint: chip.tint)
            }
            Spacer(minLength: 0)
        }
    }

    private var cta: some View {
        HStack(spacing: 4) {
            Spacer()
            Text("深入查看")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.tertiary)
            Image(systemName: "chevron.right")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.tertiary)
        }
    }

    private func trendChip(text: String, up: Bool) -> some View {
        HStack(spacing: 3) {
            Image(systemName: up ? "arrow.up.right" : "arrow.down.right")
                .font(.system(size: 9, weight: .bold))
            Text(text)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
        }
        .foregroundStyle(up ? Flux.Colors.success : Flux.Colors.warning)
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(
            (up ? Flux.Colors.success : Flux.Colors.warning).opacity(0.1),
            in: Capsule()
        )
    }

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
            .fill(Color(.secondarySystemGroupedBackground))
            .overlay(
                RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [
                                Flux.Colors.accent.opacity(0.05),
                                Color.clear
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            )
    }

    // MARK: - Loading

    /// `@MainActor`：SwiftUI View 的 private async 函数不自动继承 MainActor。
    /// 不加注解的话,`.task` 内 `await loadInsight()` 会 hop 到 global executor,
    /// 然后访问 `isLoading` / `todaySessions` (含 SwiftData @Model) → SIGABRT。
    @MainActor
    private func loadInsight() async {
        guard !isLoading else { return }
        isLoading = true
        defer { isLoading = false }

        let text: String
        if #available(iOS 26.0, *) {
            if todaySessions.isEmpty {
                text = stats.fallbackText
            } else {
                text = await BodyInsightEngine.shared.generateDailySummary(sessions: todaySessions)
            }
        } else {
            text = stats.fallbackText
        }

        insightText = text
    }
}

// MARK: - Hero Chip

private struct InsightChip: View {
    let label: String
    let value: String
    let tint: Color

    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(tint)
                .frame(width: 6, height: 6)
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundStyle(.primary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(tint.opacity(0.08), in: Capsule())
    }
}

// MARK: - Detail Sheet

struct InsightDetailView: View {
    let todaySessions: [Session]
    let recentSessions: [Session]
    let insightText: String?

    @Environment(\.dismiss) private var dismiss
    @State private var showAskCoach = false

    private var stats: InsightStats {
        InsightStats(today: todaySessions, recent: recentSessions)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 32) {
                    heroSection

                    if !todaySessions.isEmpty {
                        todayChartSection
                        signalsSection
                    }

                    if !patterns.isEmpty {
                        patternsSection
                    }

                    if recentSessions.count >= 2 {
                        weekSection
                    }

                    askCoachLink
                        .padding(.top, 8)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 24)
                .padding(.bottom, 24)
            }
            .background(Color(.systemGroupedBackground).ignoresSafeArea())
            .navigationTitle("今日洞察")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
            .sheet(isPresented: $showAskCoach) {
                AskCoachSheet(
                    todaySessions: todaySessions,
                    recentSessions: recentSessions,
                    insightText: insightText
                )
            }
        }
    }

    // MARK: - Hero

    private var heroSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(stats.headline)
                .font(.system(size: 32, weight: .semibold, design: .rounded))

            HStack(spacing: 10) {
                Text(stats.subLine)
                    .font(.system(size: 14, design: .monospaced))
                    .foregroundStyle(.secondary)
                if let delta = stats.deltaText {
                    Text(delta)
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                        .foregroundStyle(stats.deltaIsPositive ? Flux.Colors.success : Flux.Colors.warning)
                }
            }

            if let text = insightText, !text.isEmpty {
                Text(text)
                    .font(.system(size: 16))
                    .foregroundStyle(.primary)
                    .lineSpacing(7)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 10)
            }
        }
    }

    // MARK: - Today Chart

    private var todayChartSection: some View {
        section(title: "今日走势") {
            let sorted = todaySessions.sorted { $0.startedAt < $1.startedAt }
            Chart(sorted, id: \.id) { s in
                let avg = s.avgStamina ?? 0
                LineMark(x: .value("时间", s.startedAt), y: .value("续航", avg))
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(Flux.Colors.accent.gradient)
                    .lineStyle(StrokeStyle(lineWidth: 2))

                AreaMark(
                    x: .value("时间", s.startedAt),
                    yStart: .value("底", 0),
                    yEnd: .value("续航", avg)
                )
                .interpolationMethod(.catmullRom)
                .foregroundStyle(.linearGradient(
                    colors: [Flux.Colors.accent.opacity(0.18), .clear],
                    startPoint: .top,
                    endPoint: .bottom
                ))

                PointMark(x: .value("时间", s.startedAt), y: .value("续航", avg))
                    .symbolSize(40)
                    .foregroundStyle(Flux.Colors.accent)
            }
            .chartYScale(domain: 0...100)
            .chartYAxis(.hidden)
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 4)) { _ in
                    AxisValueLabel(format: .dateTime.hour().minute())
                        .font(.system(size: 9))
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(height: 140)
            .padding(.top, 4)
        }
    }

    // MARK: - Signals

    private var signalsSection: some View {
        section(title: "身体信号") {
            VStack(spacing: 12) {
                DimensionStrip(
                    label: "一致性",
                    value: stats.avgConsistency,
                    tint: Color(.systemTeal),
                    icon: "waveform.path",
                    description: stats.avgConsistency >= 0.6 ? "稳" : stats.avgConsistency >= 0.4 ? "中" : "波动"
                )
                DimensionStrip(
                    label: "紧张度",
                    value: stats.avgTension,
                    tint: Color(.systemOrange).opacity(0.85),
                    icon: "arrow.up.right",
                    description: stats.avgTension >= 0.5 ? "偏高" : stats.avgTension >= 0.3 ? "正常" : "放松"
                )
                DimensionStrip(
                    label: "疲劳度",
                    value: stats.avgFatigue,
                    tint: Color(.systemPink).opacity(0.85),
                    icon: "flame",
                    description: stats.avgFatigue >= 0.6 ? "明显" : stats.avgFatigue >= 0.3 ? "可控" : "轻微"
                )
            }
        }
    }

    // MARK: - Patterns

    private var patterns: [PatternItem] {
        var items: [PatternItem] = []
        if #available(iOS 26.0, *) {
            let anomalies = NLPSummaryEngine.shared.detectDailyAnomalies(sessions: todaySessions)
            for a in anomalies {
                items.append(PatternItem(
                    text: a.message,
                    severity: a.severity == .critical ? .high : .medium
                ))
            }
        }
        return items
    }

    private var patternsSection: some View {
        section(title: "模式") {
            VStack(spacing: 10) {
                ForEach(Array(patterns.enumerated()), id: \.offset) { _, p in
                    PatternRow(text: p.text, severity: p.severity)
                }
            }
        }
    }

    // MARK: - Week

    private var weekSection: some View {
        let cal = Calendar.current
        let grouped = Dictionary(grouping: recentSessions) { cal.startOfDay(for: $0.startedAt) }
        let days: [WeekDayPoint] = grouped.map { (date, list) in
            let avgs = list.compactMap(\.avgStamina)
            let v = avgs.isEmpty ? 0 : avgs.reduce(0, +) / Double(avgs.count)
            return WeekDayPoint(date: date, avg: v)
        }.sorted { $0.date < $1.date }

        return section(title: "近 7 天") {
            Chart(days) { d in
                BarMark(
                    x: .value("日期", d.date, unit: .day),
                    y: .value("续航", d.avg)
                )
                .foregroundStyle(Flux.Colors.accent.opacity(0.65).gradient)
                .cornerRadius(4)
            }
            .chartYScale(domain: 0...100)
            .chartYAxis(.hidden)
            .chartXAxis {
                AxisMarks(values: .stride(by: .day)) { _ in
                    AxisValueLabel(format: .dateTime.weekday(.narrow))
                        .font(.system(size: 9))
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(height: 110)
            .padding(.top, 4)
        }
    }

    // MARK: - Ask Coach (低调入口)

    private var askCoachLink: some View {
        HStack {
            Spacer()
            Button {
                showAskCoach = true
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "bubble.left.and.text.bubble.right")
                        .font(.system(size: 11))
                    Text("想问问 FocuX")
                        .font(.system(size: 12, weight: .medium))
                }
                .foregroundStyle(.tertiary)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Color(.tertiarySystemFill), in: Capsule())
            }
            .buttonStyle(.plain)
            Spacer()
        }
    }

    // MARK: - Section Wrapper

    @ViewBuilder
    private func section<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title.uppercased())
                .font(.system(size: 11, weight: .semibold))
                .tracking(1.4)
                .foregroundStyle(.tertiary)
            content()
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
                        .fill(Color(.secondarySystemGroupedBackground))
                )
        }
    }
}

// MARK: - Dimension Strip

private struct DimensionStrip: View {
    let label: String
    let value: Double           // 0...1
    let tint: Color
    let icon: String
    let description: String

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 13))
                .foregroundStyle(tint)
                .frame(width: 22)

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(label)
                        .font(.system(size: 13, weight: .medium))
                    Spacer()
                    Text("\(Int(value * 100))")
                        .font(.system(size: 13, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.primary)
                    Text(description)
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                }
                ProgressView(value: max(0, min(1, value)))
                    .tint(tint)
                    .scaleEffect(y: 0.8, anchor: .center)
            }
        }
    }
}

// MARK: - Pattern Row

private struct PatternItem {
    enum Severity { case high, medium }
    let text: String
    let severity: Severity
}

private struct PatternRow: View {
    let text: String
    let severity: PatternItem.Severity

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
                .padding(.top, 6)
            Text(text)
                .font(.system(size: 14))
                .foregroundStyle(.primary.opacity(0.85))
                .lineSpacing(4)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }

    private var color: Color {
        switch severity {
        case .high: return Flux.Colors.accent
        case .medium: return Flux.Colors.warning
        }
    }
}

// MARK: - Week Day Point

private struct WeekDayPoint: Identifiable {
    var id: Date { date }
    let date: Date
    let avg: Double
}

// MARK: - Ask Coach Sheet（聊天降级到三级，重构成更简洁的样子）

struct AskCoachSheet: View {
    let todaySessions: [Session]
    let recentSessions: [Session]
    let insightText: String?

    @Environment(\.dismiss) private var dismiss
    @State private var messages: [(question: String, answer: String)] = []
    @State private var isLoading = false
    @State private var customQuestion = ""

    private let presets: [String] = [
        "为什么我下午续航总是下降？",
        "怎样延长高效时间？",
        "我的紧张度正常吗？"
    ]

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        if messages.isEmpty {
                            presetSection
                        }
                        messageList
                        if isLoading {
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                Text("思考中…")
                                    .font(.system(size: 13))
                                    .foregroundStyle(.tertiary)
                            }
                            .padding(.horizontal, 20)
                        }
                    }
                    .padding(.vertical, 16)
                }
                .onChange(of: messages.count) { _, _ in
                    withAnimation { proxy.scrollTo(messages.count - 1, anchor: .bottom) }
                }
            }
            .background(Color(.systemGroupedBackground).ignoresSafeArea())
            .safeAreaInset(edge: .bottom) { inputBar }
            .navigationTitle("问问 FocuX")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }

    // MARK: - Presets

    private var presetSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("常见问题")
                .font(.system(size: 11, weight: .semibold))
                .tracking(1.2)
                .foregroundStyle(.tertiary)
                .padding(.horizontal, 20)

            VStack(spacing: 8) {
                ForEach(presets, id: \.self) { q in
                    Button {
                        ask(question: q)
                    } label: {
                        HStack {
                            Text(q)
                                .font(.system(size: 14))
                                .foregroundStyle(.primary)
                                .multilineTextAlignment(.leading)
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(.tertiary)
                        }
                        .padding(14)
                        .frame(maxWidth: .infinity)
                        .background(
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .fill(Color(.secondarySystemGroupedBackground))
                        )
                    }
                    .buttonStyle(.plain)
                    .disabled(isLoading)
                }
            }
            .padding(.horizontal, 20)
        }
    }

    // MARK: - Messages

    private var messageList: some View {
        VStack(alignment: .leading, spacing: 18) {
            ForEach(Array(messages.enumerated()), id: \.offset) { idx, msg in
                VStack(alignment: .leading, spacing: 10) {
                    // 用户气泡（右对齐）
                    HStack {
                        Spacer(minLength: 40)
                        Text(msg.question)
                            .font(.system(size: 14))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 9)
                            .background(Flux.Colors.accent, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    }
                    // FocuX 答（左对齐）
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "sparkles")
                            .font(.system(size: 11))
                            .foregroundStyle(Flux.Colors.accent)
                            .padding(.top, 6)
                        Text(msg.answer)
                            .font(.system(size: 14))
                            .foregroundStyle(.primary)
                            .lineSpacing(5)
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer(minLength: 0)
                    }
                    .padding(.trailing, 40)
                }
                .padding(.horizontal, 20)
                .id(idx)
            }
        }
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("输入你的问题…", text: $customQuestion)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 14))
                .submitLabel(.send)
                .onSubmit { sendCustom() }
                .disabled(isLoading)

            Button {
                sendCustom()
            } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 26))
                    .foregroundStyle(customQuestion.isEmpty ? Color(.tertiaryLabel) : Flux.Colors.accent)
            }
            .disabled(customQuestion.isEmpty || isLoading)
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }

    // MARK: - Logic

    /// `@MainActor`：SwiftUI View 的 private func 不自动继承 MainActor,
    /// `Task {}` 内访问 SwiftData @Model (sessions) 会跨 actor → SIGABRT。
    @MainActor
    private func sendCustom() {
        let q = customQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return }
        customQuestion = ""
        ask(question: q)
    }

    @MainActor
    private func ask(question: String) {
        guard !isLoading else { return }
        isLoading = true

        Task { @MainActor in
            let answer: String
            if #available(iOS 26.0, *) {
                let anomalies = NLPSummaryEngine.shared.detectDailyAnomalies(sessions: todaySessions)
                let weekly = recentSessions.count >= 2
                    ? NLPSummaryEngine.shared.generateWeeklyTrend(sessions: recentSessions)
                    : nil
                let ctx = NLPSummaryEngine.CoachContext(
                    todaySessions: todaySessions,
                    dailyInsight: insightText,
                    anomalies: anomalies,
                    weeklyTrend: weekly
                )
                answer = await NLPSummaryEngine.shared.askFollowUp(context: ctx, question: question)
            } else {
                answer = "需要 iOS 26+ 的 Apple Intelligence 才能给个性化建议。一个通用建议：每 25 分钟主动休息 5 分钟。"
            }
            messages.append((question, answer))
            isLoading = false
        }
    }
}
