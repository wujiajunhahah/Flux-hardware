import SwiftUI

/// FluxChi 的 AI 洞察呈现层 — Oura / Whoop 风格。
///
/// 入口 `DailyInsightHeroCard`：5 种 Hero variant，系统按数据状态选默认值，用户左右 swipe 切换。
/// 点击进入 `InsightDetailView`（多段 mixed-card 详情）；分享通过 `ShareCardSheet` 渲染图卡。
///
/// **拆分关系**（保持单一文件 < 400 行）：
/// - `InsightSection.swift` — 容器 + 数据派生（本文件）
/// - `InsightHeroVariants.swift` — 5 个 Hero 变体
/// - `InsightDials.swift` — `StaminaDial` / `TrendBadge` / `DotIndicator`
/// - `InsightDetailView.swift` — 详情页 + 维度行 + 模式行
/// - `InsightShareCard.swift` — 分享图卡
/// - `InsightAskCoach.swift` — 提问 sheet
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
