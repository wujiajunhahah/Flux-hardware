import SwiftUI
import Charts

// MARK: - Detail View

/// 完整洞察详情：今日 hero + 范围切换 + 多 section 卡片 + AskCoach 入口。
struct InsightDetailView: View {
    let todaySessions: [Session]
    let recentSessions: [Session]
    let insightText: String?

    @Environment(\.dismiss) private var dismiss
    @State private var range: DetailRange = .today
    @State private var showAskCoach = false
    @State private var showShare = false

    private var stats: InsightStats {
        InsightStats(today: todaySessions, recent: recentSessions)
    }

    enum DetailRange: String, CaseIterable, Identifiable {
        case today = "今日"
        case week  = "本周"
        var id: String { rawValue }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    detailHero

                    rangePicker

                    switch range {
                    case .today:  todayContent
                    case .week:   weekContent
                    }

                    askCoachLink
                        .padding(.top, 8)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 20)
                .padding(.bottom, 24)
            }
            .background(Color(.systemGroupedBackground).ignoresSafeArea())
            .navigationTitle("今日洞察")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
                ToolbarItem(placement: .topBarLeading) {
                    if stats.hasData {
                        Button {
                            showShare = true
                        } label: {
                            Image(systemName: "square.and.arrow.up")
                        }
                    }
                }
            }
            .sheet(isPresented: $showAskCoach) {
                AskCoachSheet(
                    todaySessions: todaySessions,
                    recentSessions: recentSessions,
                    insightText: insightText
                )
            }
            .sheet(isPresented: $showShare) {
                ShareCardSheet(stats: stats, narrative: insightText)
            }
        }
    }

    // MARK: Hero

    private var detailHero: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 16) {
                StaminaDial(value: stats.avgStamina, tint: Flux.Colors.forStaminaValue(stats.avgStamina))
                    .frame(width: 110, height: 110)

                VStack(alignment: .leading, spacing: 6) {
                    Text(stats.headline)
                        .font(.system(size: 24, weight: .semibold, design: .rounded))
                    Text(stats.subLine)
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundStyle(.secondary)
                    if let delta = stats.deltaVsYesterday, abs(delta) >= 3 {
                        TrendBadge(delta: delta).padding(.top, 2)
                    }
                    Spacer(minLength: 0)
                }
            }

            if let text = insightText, !text.isEmpty {
                Text(text)
                    .font(.system(size: 15))
                    .foregroundStyle(.primary)
                    .lineSpacing(6)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 6)
            }
        }
    }

    private var rangePicker: some View {
        Picker("范围", selection: $range) {
            ForEach(DetailRange.allCases) { r in
                Text(r.rawValue).tag(r)
            }
        }
        .pickerStyle(.segmented)
    }

    // MARK: Today content

    @ViewBuilder
    private var todayContent: some View {
        if todaySessions.isEmpty {
            emptyTodayCard
        } else {
            todayChartCard
            signalsCard
            if !patterns.isEmpty {
                patternsCard
            }
            recommendationCard
        }
    }

    private var emptyTodayCard: some View {
        sectionCard(title: "还没有今日数据") {
            HStack(spacing: 12) {
                Image(systemName: "sparkles")
                    .font(.system(size: 24))
                    .foregroundStyle(Flux.Colors.accent.opacity(0.5))
                Text(stats.emptyInvitation)
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
                Spacer()
            }
        }
    }

    private var todayChartCard: some View {
        sectionCard(title: "走势") {
            let sorted = todaySessions.sorted { $0.startedAt < $1.startedAt }
            Chart(sorted, id: \.id) { s in
                let avg = s.avgStamina ?? 0
                LineMark(x: .value("时间", s.startedAt), y: .value("续航", avg))
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(Flux.Colors.forStaminaValue(stats.avgStamina).gradient)
                    .lineStyle(StrokeStyle(lineWidth: 2))

                AreaMark(
                    x: .value("时间", s.startedAt),
                    yStart: .value("底", 0),
                    yEnd: .value("续航", avg)
                )
                .interpolationMethod(.catmullRom)
                .foregroundStyle(.linearGradient(
                    colors: [Flux.Colors.accent.opacity(0.18), .clear],
                    startPoint: .top, endPoint: .bottom
                ))

                PointMark(x: .value("时间", s.startedAt), y: .value("续航", avg))
                    .symbolSize(36)
                    .foregroundStyle(Flux.Colors.forStaminaValue(avg))
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
        }
    }

    private var signalsCard: some View {
        sectionCard(title: "身体信号") {
            VStack(spacing: 16) {
                DimensionSparklineRow(
                    label: "一致性",
                    value: stats.avgConsistency,
                    tint: Color(.systemTeal),
                    icon: "waveform.path",
                    description: stats.avgConsistency >= 0.6 ? "稳定" : stats.avgConsistency >= 0.4 ? "中等" : "波动"
                )
                DimensionSparklineRow(
                    label: "紧张度",
                    value: stats.avgTension,
                    tint: Color(.systemOrange).opacity(0.85),
                    icon: "arrow.up.right",
                    description: stats.avgTension >= 0.5 ? "偏高" : stats.avgTension >= 0.3 ? "正常" : "放松"
                )
                DimensionSparklineRow(
                    label: "疲劳度",
                    value: stats.avgFatigue,
                    tint: Color(.systemPink).opacity(0.85),
                    icon: "flame",
                    description: stats.avgFatigue >= 0.6 ? "明显" : stats.avgFatigue >= 0.3 ? "可控" : "轻微"
                )
            }
        }
    }

    private var patternsCard: some View {
        sectionCard(title: "模式") {
            VStack(alignment: .leading, spacing: 12) {
                ForEach(Array(patterns.enumerated()), id: \.offset) { _, p in
                    PatternRow(text: p.text, severity: p.severity)
                }
            }
        }
    }

    private var recommendationCard: some View {
        sectionCard(title: "建议") {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: tipIcon)
                    .font(.system(size: 16))
                    .foregroundStyle(Flux.Colors.accent)
                    .frame(width: 22)
                Text(tipText)
                    .font(.system(size: 14))
                    .foregroundStyle(.primary)
                    .lineSpacing(4)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer()
            }
        }
    }

    private var tipIcon: String {
        if stats.avgFatigue > 0.5 { return "moon.zzz" }
        if stats.avgTension > 0.5 { return "figure.cooldown" }
        if stats.avgStamina < 50 { return "leaf" }
        return "sparkle"
    }

    private var tipText: String {
        if stats.avgFatigue > 0.5 {
            return "疲劳偏高。下一段时长缩到 20 分钟，结束后做一次 5 分钟拉伸。"
        }
        if stats.avgTension > 0.5 {
            return "肩颈紧张持续偏高。把屏幕抬高到视线平齐，每 30 分钟做一次肩部画圈。"
        }
        if stats.avgStamina < 50 {
            return "续航偏低，今天可以提前收尾。明天试试在上午第一段处理最难的任务。"
        }
        return "节奏稳。下一段可以挑战 25 分钟以上，结束后再判断是否延长。"
    }

    // MARK: Week content

    @ViewBuilder
    private var weekContent: some View {
        if recentSessions.count < 2 {
            sectionCard(title: "本周") {
                HStack(spacing: 12) {
                    Image(systemName: "calendar")
                        .font(.system(size: 24))
                        .foregroundStyle(.tertiary)
                    Text("还需要至少 2 次记录才能解锁本周分析。")
                        .font(.system(size: 14))
                        .foregroundStyle(.secondary)
                    Spacer()
                }
            }
        } else {
            weeklyBarsCard
            weeklySlotCard
            weeklyHighlightCard
        }
    }

    private var weeklyBarsCard: some View {
        let cal = Calendar.current
        let grouped = Dictionary(grouping: recentSessions) { cal.startOfDay(for: $0.startedAt) }
        let days: [WeekDayPoint] = grouped.map { (date, list) in
            let avgs = list.compactMap(\.avgStamina)
            let v = avgs.isEmpty ? 0 : avgs.reduce(0, +) / Double(avgs.count)
            return WeekDayPoint(date: date, avg: v)
        }.sorted { $0.date < $1.date }

        return sectionCard(title: "近 7 天") {
            Chart(days) { d in
                BarMark(
                    x: .value("日期", d.date, unit: .day),
                    y: .value("续航", d.avg)
                )
                .foregroundStyle(Flux.Colors.forStaminaValue(d.avg).gradient)
                .cornerRadius(5)
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
            .frame(height: 120)
        }
    }

    private var weeklySlotCard: some View {
        var slotSums: [Flux.TimeSlot: (sum: Double, n: Int)] = [:]
        for s in recentSessions {
            guard let v = s.avgStamina else { continue }
            let slot = Flux.TimeSlot.from(date: s.startedAt)
            let cur = slotSums[slot] ?? (0, 0)
            slotSums[slot] = (cur.sum + v, cur.n + 1)
        }
        let rows = slotSums
            .map { (slot: $0.key, avg: $0.value.sum / Double($0.value.n), count: $0.value.n) }
            .sorted { $0.slot.order < $1.slot.order }

        return sectionCard(title: "时段分布") {
            VStack(spacing: 10) {
                ForEach(rows, id: \.slot) { row in
                    HStack(spacing: 10) {
                        Image(systemName: row.slot.iconName)
                            .font(.system(size: 11))
                            .foregroundStyle(Flux.Colors.forStaminaValue(row.avg))
                            .frame(width: 16)
                        Text(row.slot.rawValue)
                            .font(.system(size: 12, weight: .medium))
                        Spacer()
                        Text("\(row.count) 段")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.tertiary)
                        Text("\(Int(row.avg))")
                            .font(.system(size: 13, weight: .semibold, design: .monospaced))
                            .foregroundStyle(Flux.Colors.forStaminaValue(row.avg))
                            .frame(width: 28, alignment: .trailing)
                    }
                }
            }
        }
    }

    private var weeklyHighlightCard: some View {
        sectionCard(title: "本周") {
            VStack(alignment: .leading, spacing: 8) {
                if let weekly = stats.weeklyAvg {
                    HStack(alignment: .firstTextBaseline, spacing: 4) {
                        Text("\(Int(weekly))")
                            .font(.system(size: 30, weight: .bold, design: .rounded))
                        Text("/ 100")
                            .font(.system(size: 12))
                            .foregroundStyle(.tertiary)
                        Spacer()
                    }
                }
                if let best = stats.bestSlot {
                    Text("\(best.0.rawValue) 平均续航 \(Int(best.1))，是本周最佳时段。")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                } else {
                    Text("过去 7 天的工作节奏。")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: Patterns

    private struct PatternItemLocal {
        let text: String
        let severity: PatternRow.PatternSeverity
    }

    private var patterns: [PatternItemLocal] {
        var items: [PatternItemLocal] = []
        if #available(iOS 26.0, *) {
            let anomalies = NLPSummaryEngine.shared.detectDailyAnomalies(sessions: todaySessions)
            for a in anomalies {
                items.append(PatternItemLocal(
                    text: a.message,
                    severity: a.severity == .critical ? .high : .medium
                ))
            }
        }
        return items
    }

    // MARK: AskCoach link

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

    // MARK: Section card wrapper

    @ViewBuilder
    private func sectionCard<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
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

// MARK: - DimensionSparklineRow

/// 单维度信号行：图标 + label + 10 段填充条 + 数值。
struct DimensionSparklineRow: View {
    let label: String
    let value: Double  // 0...1
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
                    Text(description)
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                    Text("\(Int(value * 100))")
                        .font(.system(size: 14, weight: .semibold, design: .monospaced))
                        .foregroundStyle(tint)
                        .contentTransition(.numericText(value: value))
                        .frame(width: 30, alignment: .trailing)
                }
                segmentedBar
            }
        }
    }

    private var segmentedBar: some View {
        GeometryReader { geo in
            let segments = 10
            let gap: CGFloat = 3
            let totalGap = CGFloat(segments - 1) * gap
            let segW = (geo.size.width - totalGap) / CGFloat(segments)
            let fillCount = Int((max(0, min(1, value)) * Double(segments)).rounded(.up))

            HStack(spacing: gap) {
                ForEach(0..<segments, id: \.self) { i in
                    RoundedRectangle(cornerRadius: 2)
                        .fill(i < fillCount ? tint : tint.opacity(0.13))
                        .frame(width: segW, height: 7)
                }
            }
        }
        .frame(height: 7)
    }
}

// MARK: - Pattern Row

/// 模式洞察行：彩色点 + 多行文本。
struct PatternRow: View {
    let text: String
    let severity: PatternSeverity

    enum PatternSeverity { case high, medium }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
                .padding(.top, 6)
            Text(text)
                .font(.system(size: 14))
                .foregroundStyle(.primary)
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

// MARK: - WeekDayPoint (Chart data)

struct WeekDayPoint: Identifiable {
    var id: Date { date }
    let date: Date
    let avg: Double
}
