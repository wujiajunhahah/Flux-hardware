import SwiftUI
import Charts

/// FluxChi 的 AI 洞察呈现层 — Oura / Whoop 风格。
///
/// Hero 区有 5 种 variant，系统按数据状态默认选一个，用户左右 swipe 切换；
/// Detail 是策划过的多段 mixed-card 布局；分享按钮通过 `ImageRenderer` 输出今日图卡。
///
/// **Actor 安全**：所有访问 SwiftData `@Model` 的方法都在 MainActor 上；
/// Foundation Models 内部 `await session.respond(to:)` 自动 hop 到后台跑推理。

// MARK: - Hero Variant 枚举

enum HeroVariant: String, CaseIterable, Hashable {
    case streak    // 默认 — 大数字 + dial + trend
    case quote     // narrative 主视觉
    case compare   // 今 vs 昨
    case empty     // 还没数据
    case critical  // 严重异常

    var title: String {
        switch self {
        case .streak:   return "数字"
        case .quote:    return "观察"
        case .compare:  return "对比"
        case .empty:    return "开始"
        case .critical: return "关注"
        }
    }

    var icon: String {
        switch self {
        case .streak:   return "bolt.fill"
        case .quote:    return "quote.opening"
        case .compare:  return "chart.bar.xaxis"
        case .empty:    return "moon.stars"
        case .critical: return "exclamationmark.circle"
        }
    }
}

// MARK: - InsightStats

struct InsightStats {
    let today: [Session]
    let recent: [Session]

    // MARK: Aggregates

    var avgStamina: Double {
        let v = today.compactMap(\.avgStamina)
        guard !v.isEmpty else { return 0 }
        return v.reduce(0, +) / Double(v.count)
    }

    var totalMinutes: Int {
        Int(today.reduce(0) { $0 + $1.duration } / 60)
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

    var staminaSparkline: [Double] {
        today.sorted { $0.startedAt < $1.startedAt }.compactMap(\.avgStamina)
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

    var weeklyAvg: Double? {
        let v = recent.compactMap(\.avgStamina)
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

    // MARK: 状态判定

    var hasData: Bool { !today.isEmpty }

    var isCritical: Bool {
        guard hasData else { return false }
        if avgStamina < 30 { return true }
        if avgFatigue > 0.7 { return true }
        return false
    }

    var deltaVsYesterday: Int? {
        guard hasData, let prev = previousDayAvg else { return nil }
        return Int(avgStamina - prev)
    }

    var availableVariants: [HeroVariant] {
        if !hasData {
            return [.empty, .quote]
        }
        var list: [HeroVariant] = [.streak, .quote]
        if previousDayAvg != nil {
            list.insert(.compare, at: 1)
        }
        if isCritical {
            list.insert(.critical, at: 0)
        }
        return list
    }

    var defaultVariant: HeroVariant {
        if !hasData { return .empty }
        if isCritical { return .critical }
        return .streak
    }

    // MARK: Headline / Sub

    var headline: String {
        if !hasData {
            let hour = Calendar.current.component(.hour, from: Date())
            if hour < 12  { return "新的一天" }
            if hour < 18  { return "今天还在等待" }
            return "今晚静下来"
        }
        if isCritical { return "身体在说慢一点" }
        let s = avgStamina
        if s >= 75 { return "今天身体在线" }
        if s >= 60 { return "状态在调整" }
        if s >= 40 { return "今天有起伏" }
        return "需要休息"
    }

    var subLine: String {
        if !hasData { return "暂无今日数据" }
        return "\(today.count) 段 · \(totalMinutes) 分钟"
    }

    var emptyInvitation: String {
        let hour = Calendar.current.component(.hour, from: Date())
        if hour < 12 { return "戴上手环，开启第一段专注。" }
        if hour < 18 { return "找一个 25 分钟，让身体说话。" }
        return "今晚不用勉强，明天再继续。"
    }

    var fallbackQuote: String {
        if !hasData { return emptyInvitation }
        if isCritical {
            return "续航偏低，肌肉信号在反复发出疲劳。今天可以早点收尾。"
        }
        let mins = totalMinutes
        if avgStamina >= 75 {
            return "今天 \(today.count) 段专注 \(mins) 分钟，续航 \(Int(avgStamina))，节奏很稳。"
        }
        return "\(today.count) 段共 \(mins) 分钟，续航 \(Int(avgStamina))，有空间再优化休息节奏。"
    }
}

// MARK: - DailyInsightHeroCard (Container)

struct DailyInsightHeroCard: View {
    let todaySessions: [Session]
    let recentSessions: [Session]

    @State private var insightText: String?
    @State private var isLoading = false
    @State private var showDetail = false
    @State private var selectedVariant: HeroVariant
    @State private var showShare = false

    private var stats: InsightStats {
        InsightStats(today: todaySessions, recent: recentSessions)
    }

    init(todaySessions: [Session], recentSessions: [Session]) {
        self.todaySessions = todaySessions
        self.recentSessions = recentSessions
        let s = InsightStats(today: todaySessions, recent: recentSessions)
        self._selectedVariant = State(initialValue: s.defaultVariant)
    }

    var body: some View {
        VStack(spacing: 12) {
            heroCarousel
            if stats.availableVariants.count > 1 {
                DotIndicator(
                    count: stats.availableVariants.count,
                    selected: stats.availableVariants.firstIndex(of: selectedVariant) ?? 0
                )
                .padding(.top, -4)
            }
        }
        .contextMenu {
            Section("布局") {
                ForEach(stats.availableVariants, id: \.self) { v in
                    Button {
                        withAnimation(.spring(response: 0.45, dampingFraction: 0.85)) {
                            selectedVariant = v
                        }
                    } label: {
                        Label(v.title, systemImage: v.icon)
                    }
                }
            }
            Button {
                showDetail = true
            } label: {
                Label("查看完整洞察", systemImage: "rectangle.expand.vertical")
            }
            if stats.hasData {
                Button {
                    showShare = true
                } label: {
                    Label("分享今日卡片", systemImage: "square.and.arrow.up")
                }
            }
        }
        .sheet(isPresented: $showDetail) {
            InsightDetailView(
                todaySessions: todaySessions,
                recentSessions: recentSessions,
                insightText: insightText
            )
        }
        .sheet(isPresented: $showShare) {
            ShareCardSheet(stats: stats, narrative: insightText)
        }
        .onChange(of: todaySessions.count) { _, _ in
            if !stats.availableVariants.contains(selectedVariant) {
                selectedVariant = stats.defaultVariant
            }
        }
        .task(id: todaySessions.count) {
            await loadInsight()
        }
    }

    // MARK: - Carousel

    private var heroCarousel: some View {
        TabView(selection: $selectedVariant) {
            ForEach(stats.availableVariants, id: \.self) { variant in
                heroVariantView(variant)
                    .tag(variant)
            }
        }
        .tabViewStyle(.page(indexDisplayMode: .never))
        .frame(height: 230)
        .animation(.spring(response: 0.45, dampingFraction: 0.85), value: selectedVariant)
    }

    @ViewBuilder
    private func heroVariantView(_ variant: HeroVariant) -> some View {
        switch variant {
        case .streak:
            HeroStreakView(stats: stats, onTap: { showDetail = true })
        case .quote:
            HeroQuoteView(stats: stats, narrative: insightText, isLoading: isLoading, onTap: { showDetail = true })
        case .compare:
            HeroCompareView(stats: stats, onTap: { showDetail = true })
        case .empty:
            HeroEmptyView(stats: stats, onTap: { showDetail = true })
        case .critical:
            HeroCriticalView(stats: stats, narrative: insightText, onTap: { showDetail = true })
        }
    }

    // MARK: - Loading

    @MainActor
    private func loadInsight() async {
        guard !isLoading else { return }
        isLoading = true
        defer { isLoading = false }

        let text: String
        if #available(iOS 26.0, *) {
            if todaySessions.isEmpty {
                text = stats.fallbackQuote
            } else {
                text = await BodyInsightEngine.shared.generateDailySummary(sessions: todaySessions)
            }
        } else {
            text = stats.fallbackQuote
        }
        insightText = text
    }
}

// MARK: - Hero/Streak

struct HeroStreakView: View {
    let stats: InsightStats
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 18) {
                // Left: dial + big number
                VStack(alignment: .leading, spacing: 8) {
                    StaminaDial(
                        value: stats.avgStamina,
                        tint: tint
                    )
                    .frame(width: 96, height: 96)

                    Text(stats.headline)
                        .font(.system(size: 17, weight: .semibold, design: .rounded))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                }

                // Right: sub data
                VStack(alignment: .leading, spacing: 14) {
                    metric(icon: "timer", label: "今日累计", value: "\(stats.totalMinutes) 分钟")
                    metric(icon: "rectangle.stack.fill", label: "场次", value: "\(stats.today.count)")
                    if let delta = stats.deltaVsYesterday, abs(delta) >= 3 {
                        TrendBadge(delta: delta)
                    }
                    if let best = stats.bestSlot, best.1 > 50 {
                        chip(icon: best.0.iconName, text: "\(best.0.rawValue)最佳", tint: Flux.Colors.success)
                    }
                    Spacer(minLength: 0)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(cardBackground)
        }
        .buttonStyle(.plain)
    }

    private var tint: Color {
        Flux.Colors.forStaminaValue(stats.avgStamina)
    }

    private func metric(icon: String, label: String, value: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
                .frame(width: 14)
            Text(label)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
            Spacer(minLength: 0)
            Text(value)
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundStyle(.primary)
                .contentTransition(.numericText())
        }
    }

    private func chip(icon: String, text: String, tint: Color) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 10))
            Text(text)
                .font(.system(size: 11, weight: .medium))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(tint.opacity(0.10), in: Capsule())
    }

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
            .fill(Color(.secondarySystemGroupedBackground))
            .overlay(
                RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [tint.opacity(0.06), .clear],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            )
    }
}

// MARK: - Hero/Quote

struct HeroQuoteView: View {
    let stats: InsightStats
    let narrative: String?
    let isLoading: Bool
    let onTap: () -> Void

    private var text: String { narrative ?? stats.fallbackQuote }

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Image(systemName: "quote.opening")
                        .font(.system(size: 26, weight: .semibold))
                        .foregroundStyle(Flux.Colors.accent.opacity(0.7))
                    Spacer()
                    if isLoading {
                        ProgressView().controlSize(.small)
                    }
                }
                Text(text)
                    .font(.system(size: 17, weight: .medium))
                    .foregroundStyle(.primary)
                    .lineSpacing(6)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
                HStack(spacing: 6) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 10))
                    Text(stats.hasData ? "FocuX 今日观察" : "FocuX")
                        .font(.system(size: 11, weight: .medium))
                }
                .foregroundStyle(.tertiary)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(quoteBackground)
        }
        .buttonStyle(.plain)
    }

    private var quoteBackground: some View {
        RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
            .fill(Color(.secondarySystemGroupedBackground))
    }
}

// MARK: - Hero/Compare

struct HeroCompareView: View {
    let stats: InsightStats
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 6) {
                    Image(systemName: "chart.bar.xaxis")
                        .font(.system(size: 11))
                    Text("今 vs 昨")
                        .font(.system(size: 11, weight: .semibold))
                        .tracking(0.5)
                }
                .foregroundStyle(.tertiary)

                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("\(Int(stats.avgStamina))")
                        .font(.system(size: 56, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)
                        .contentTransition(.numericText(value: stats.avgStamina))
                    if let delta = stats.deltaVsYesterday {
                        TrendBadge(delta: delta)
                            .padding(.bottom, 6)
                    }
                }
                .padding(.vertical, -4)

                bars
                Spacer(minLength: 0)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
            )
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var bars: some View {
        let yesterday = stats.previousDayAvg ?? 0
        let today = stats.avgStamina
        let maxV = max(max(yesterday, today), 1)

        VStack(spacing: 10) {
            comparisonRow(label: "昨天", value: yesterday, ratio: yesterday / maxV, tint: .gray)
            comparisonRow(label: "今天", value: today, ratio: today / maxV, tint: Flux.Colors.forStaminaValue(today))
        }
    }

    private func comparisonRow(label: String, value: Double, ratio: Double, tint: Color) -> some View {
        HStack(spacing: 10) {
            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: 32, alignment: .leading)

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 5)
                        .fill(tint.opacity(0.12))
                        .frame(height: 14)
                    RoundedRectangle(cornerRadius: 5)
                        .fill(tint.gradient)
                        .frame(width: geo.size.width * max(0.04, ratio), height: 14)
                }
            }
            .frame(height: 14)

            Text("\(Int(value))")
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundStyle(.primary)
                .frame(width: 32, alignment: .trailing)
        }
    }
}

// MARK: - Hero/Empty

struct HeroEmptyView: View {
    let stats: InsightStats
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 14) {
                Image(systemName: "moon.stars")
                    .font(.system(size: 30, weight: .light))
                    .foregroundStyle(Flux.Colors.accent.opacity(0.6))
                    .symbolEffect(.pulse.byLayer, options: .repeating)

                Text(stats.headline)
                    .font(.system(size: 28, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary)

                Text(stats.emptyInvitation)
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
                    .lineSpacing(4)
                    .fixedSize(horizontal: false, vertical: true)

                Spacer(minLength: 0)

                if let weekly = stats.weeklyAvg, weekly > 0 {
                    HStack(spacing: 8) {
                        Image(systemName: "calendar.badge.checkmark")
                            .font(.system(size: 10))
                        Text("过去 7 天平均续航 \(Int(weekly))")
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                    }
                    .foregroundStyle(.tertiary)
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Hero/Critical

struct HeroCriticalView: View {
    let stats: InsightStats
    let narrative: String?
    let onTap: () -> Void

    private var text: String {
        narrative ?? stats.fallbackQuote
    }

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.circle.fill")
                        .font(.system(size: 14))
                        .foregroundStyle(Flux.Colors.warning)
                        .symbolEffect(.pulse, options: .repeating)
                    Text("需要关注")
                        .font(.system(size: 12, weight: .semibold))
                        .tracking(0.5)
                        .foregroundStyle(Flux.Colors.warning)
                    Spacer()
                    Text("\(Int(stats.avgStamina))")
                        .font(.system(size: 22, weight: .bold, design: .rounded))
                        .foregroundStyle(Flux.Colors.warning)
                        .contentTransition(.numericText(value: stats.avgStamina))
                }

                Text(text)
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(.primary)
                    .lineSpacing(5)
                    .fixedSize(horizontal: false, vertical: true)

                Spacer(minLength: 0)

                HStack(spacing: 8) {
                    if stats.avgFatigue > 0.5 {
                        signalChip(label: "疲劳 \(Int(stats.avgFatigue * 100))%", tint: Flux.Colors.warning)
                    }
                    if stats.avgTension > 0.5 {
                        signalChip(label: "紧张 \(Int(stats.avgTension * 100))%", tint: Color(.systemOrange))
                    }
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
                    .overlay(
                        RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
                            .stroke(Flux.Colors.warning.opacity(0.3), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(.plain)
    }

    private func signalChip(label: String, tint: Color) -> some View {
        Text(label)
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(tint)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(tint.opacity(0.12), in: Capsule())
    }
}

// MARK: - StaminaDial (Canvas)

struct StaminaDial: View {
    let value: Double  // 0...100
    let tint: Color

    private var progress: Double { max(0, min(1, value / 100)) }

    var body: some View {
        Canvas { context, size in
            let lineWidth: CGFloat = 9
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let radius = min(size.width, size.height) / 2 - lineWidth / 2 - 2

            // Background track
            let bg = Path { p in
                p.addArc(
                    center: center,
                    radius: radius,
                    startAngle: .degrees(-90),
                    endAngle: .degrees(270),
                    clockwise: false
                )
            }
            context.stroke(bg, with: .color(tint.opacity(0.12)),
                           style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))

            // Progress
            if progress > 0 {
                let endAngle = -90.0 + 360.0 * progress
                let fg = Path { p in
                    p.addArc(
                        center: center,
                        radius: radius,
                        startAngle: .degrees(-90),
                        endAngle: .degrees(endAngle),
                        clockwise: false
                    )
                }
                let gradient = Gradient(colors: [tint.opacity(0.75), tint])
                context.stroke(
                    fg,
                    with: .conicGradient(
                        gradient,
                        center: center,
                        angle: .degrees(-90)
                    ),
                    style: StrokeStyle(lineWidth: lineWidth, lineCap: .round)
                )
            }
        }
        .overlay {
            VStack(spacing: 0) {
                Text("\(Int(value))")
                    .font(.system(size: 32, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)
                    .contentTransition(.numericText(value: value))
                Text("续航")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.tertiary)
                    .padding(.top, -2)
            }
        }
        .animation(.spring(response: 0.6, dampingFraction: 0.85), value: value)
    }
}

// MARK: - TrendBadge

struct TrendBadge: View {
    let delta: Int

    private var up: Bool { delta >= 0 }
    private var absText: String { delta > 0 ? "+\(delta)" : "\(delta)" }

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: up ? "arrow.up.right" : "arrow.down.right")
                .font(.system(size: 9, weight: .bold))
            Text(absText)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
        }
        .foregroundStyle(up ? Flux.Colors.success : Flux.Colors.warning)
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(
            (up ? Flux.Colors.success : Flux.Colors.warning).opacity(0.12),
            in: Capsule()
        )
    }
}

// MARK: - DotIndicator

struct DotIndicator: View {
    let count: Int
    let selected: Int

    var body: some View {
        HStack(spacing: 6) {
            ForEach(0..<count, id: \.self) { idx in
                Capsule()
                    .fill(idx == selected ? Color.primary.opacity(0.8) : Color.primary.opacity(0.15))
                    .frame(width: idx == selected ? 18 : 6, height: 6)
                    .animation(.spring(response: 0.4, dampingFraction: 0.8), value: selected)
            }
        }
    }
}

// MARK: - Detail View

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

// MARK: - WeekDayPoint

private struct WeekDayPoint: Identifiable {
    var id: Date { date }
    let date: Date
    let avg: Double
}

// MARK: - Share Card

struct ShareCardSheet: View {
    let stats: InsightStats
    let narrative: String?

    @Environment(\.dismiss) private var dismiss
    @State private var renderedImage: Image?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    cardPreview
                        .padding(.horizontal, 24)
                        .padding(.top, 16)

                    if let img = renderedImage {
                        ShareLink(
                            item: img,
                            preview: SharePreview("FocuX 今日洞察", image: img)
                        ) {
                            HStack(spacing: 8) {
                                Image(systemName: "square.and.arrow.up")
                                Text("分享")
                                    .font(.system(size: 16, weight: .semibold))
                            }
                            .foregroundStyle(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Flux.Colors.accent, in: Capsule())
                        }
                    } else {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small)
                            Text("生成图片中…")
                                .font(.system(size: 13))
                                .foregroundStyle(.tertiary)
                        }
                    }
                }
                .padding(.bottom, 32)
            }
            .background(Color(.systemGroupedBackground).ignoresSafeArea())
            .navigationTitle("分享今日卡片")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
        .onAppear {
            render()
        }
    }

    @ViewBuilder
    private var cardPreview: some View {
        ShareCardContent(stats: stats, narrative: narrative)
            .shadow(color: .black.opacity(0.15), radius: 16, y: 6)
    }

    @MainActor
    private func render() {
        let renderer = ImageRenderer(content: ShareCardContent(stats: stats, narrative: narrative).frame(width: 380))
        renderer.scale = UIScreen.main.scale
        if let ui = renderer.uiImage {
            renderedImage = Image(uiImage: ui)
        }
    }
}

struct ShareCardContent: View {
    let stats: InsightStats
    let narrative: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                HStack(spacing: 6) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 11, weight: .semibold))
                    Text("FocuX 今日")
                        .font(.system(size: 11, weight: .semibold))
                        .tracking(0.8)
                }
                .foregroundStyle(.white.opacity(0.85))
                Spacer()
                Text(Date().formatted(.dateTime.month().day()))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.7))
            }

            HStack(alignment: .top, spacing: 16) {
                ZStack {
                    Circle()
                        .stroke(Color.white.opacity(0.2), lineWidth: 7)
                    Circle()
                        .trim(from: 0, to: max(0.04, min(1, stats.avgStamina / 100)))
                        .stroke(Color.white, style: StrokeStyle(lineWidth: 7, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                    Text("\(Int(stats.avgStamina))")
                        .font(.system(size: 30, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                }
                .frame(width: 100, height: 100)

                VStack(alignment: .leading, spacing: 8) {
                    Text(stats.headline)
                        .font(.system(size: 22, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white)
                    Text(stats.subLine)
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.75))
                    if let delta = stats.deltaVsYesterday, abs(delta) >= 3 {
                        Text(delta > 0 ? "比昨天 +\(delta)" : "比昨天 \(delta)")
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.85))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color.white.opacity(0.15), in: Capsule())
                    }
                    Spacer(minLength: 0)
                }
            }

            if let text = narrative ?? Optional(stats.fallbackQuote), !text.isEmpty {
                Text(text)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.white.opacity(0.9))
                    .lineSpacing(5)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 14) {
                shareMetric(label: "紧张", value: "\(Int(stats.avgTension * 100))%")
                shareMetric(label: "疲劳", value: "\(Int(stats.avgFatigue * 100))%")
                shareMetric(label: "一致性", value: "\(Int(stats.avgConsistency * 100))%")
            }

            HStack {
                Text("focux.me")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.white.opacity(0.5))
                Spacer()
                Text("EMG · Vision · Focus")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
        .padding(22)
        .frame(maxWidth: .infinity)
        .background(
            LinearGradient(
                colors: [
                    Color.black,
                    Flux.Colors.accent.opacity(0.55)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 28, style: .continuous)
        )
    }

    private func shareMetric(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .tracking(0.5)
                .foregroundStyle(.white.opacity(0.6))
            Text(value)
                .font(.system(size: 16, weight: .semibold, design: .monospaced))
                .foregroundStyle(.white)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - AskCoachSheet (polished)

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

    private var presetSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("常见问题")
                .font(.system(size: 11, weight: .semibold))
                .tracking(1.2)
                .foregroundStyle(.tertiary)
                .padding(.horizontal, 20)
            VStack(spacing: 8) {
                ForEach(presets, id: \.self) { q in
                    Button { ask(question: q) } label: {
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

    private var messageList: some View {
        VStack(alignment: .leading, spacing: 18) {
            ForEach(Array(messages.enumerated()), id: \.offset) { idx, msg in
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Spacer(minLength: 40)
                        Text(msg.question)
                            .font(.system(size: 14))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 9)
                            .background(Flux.Colors.accent, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    }
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
                .transition(.opacity.combined(with: .move(edge: .bottom)))
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
            Button { sendCustom() } label: {
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
            withAnimation(.spring(response: 0.45, dampingFraction: 0.85)) {
                messages.append((question, answer))
            }
            isLoading = false
        }
    }
}
