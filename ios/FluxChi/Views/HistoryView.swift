import SwiftUI
import SwiftData

/// Oura/Whoop 风格的历史页：周英雄卡 + 28 天趋势 + 按天分组的 session 卡片。
/// 原生 iOS HIG：`Charts`、`.ultraThinMaterial`、systemGroupedBackground、large title。
struct HistoryView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Session.startedAt, order: .reverse) private var sessions: [Session]

    @State private var searchText = ""
    @State private var exportError: String?
    @State private var showExportError = false

    // MARK: - Derived

    private var completedSessions: [Session] {
        sessions.filter { !$0.isActive }
    }

    private var filtered: [Session] {
        guard !searchText.isEmpty else { return completedSessions }
        return completedSessions.filter {
            $0.title.localizedCaseInsensitiveContains(searchText)
        }
    }

    private var dayGroups: [(date: Date, sessions: [Session])] {
        let cal = Calendar.current
        let dict = Dictionary(grouping: filtered) { cal.startOfDay(for: $0.startedAt) }
        return dict.map { ($0.key, $0.value.sorted { $0.startedAt > $1.startedAt }) }
            .sorted { $0.date > $1.date }
    }

    // 当前周 (Mon–Sun)
    private var weekStart: Date {
        let cal = Calendar(identifier: .iso8601)
        let comps = cal.dateComponents([.yearForWeekOfYear, .weekOfYear], from: Date())
        return cal.date(from: comps) ?? Date()
    }

    private var thisWeekSessions: [Session] {
        completedSessions.filter { $0.startedAt >= weekStart }
    }

    private var weekBars: [HistoryWeekHero.DayBar] {
        let cal = Calendar.current
        let today = cal.startOfDay(for: Date())
        let weekdayLabels = ["一", "二", "三", "四", "五", "六", "日"]
        return (0..<7).map { offset in
            let date = cal.date(byAdding: .day, value: offset, to: weekStart) ?? weekStart
            let day = cal.startOfDay(for: date)
            let sessionsOfDay = completedSessions.filter { cal.isDate($0.startedAt, inSameDayAs: day) }
            let totalMin = sessionsOfDay.reduce(0.0) { $0 + $1.duration / 60 }
            let avgVals = sessionsOfDay.compactMap(\.avgStamina)
            let avg = avgVals.isEmpty ? 0 : avgVals.reduce(0, +) / Double(avgVals.count)
            return HistoryWeekHero.DayBar(
                date: day,
                weekdayShort: weekdayLabels[offset],
                totalMinutes: totalMin,
                avgStamina: avg,
                isToday: cal.isDate(day, inSameDayAs: today)
            )
        }
    }

    private var trendPoints: [HistoryTrendCard.DayPoint] {
        let cal = Calendar.current
        let today = cal.startOfDay(for: Date())
        return (0..<28).reversed().map { offset in
            let day = cal.date(byAdding: .day, value: -offset, to: today) ?? today
            let sessionsOfDay = completedSessions.filter { cal.isDate($0.startedAt, inSameDayAs: day) }
            let avgVals = sessionsOfDay.compactMap(\.avgStamina)
            let avg = avgVals.isEmpty ? 0 : avgVals.reduce(0, +) / Double(avgVals.count)
            return HistoryTrendCard.DayPoint(date: day, avgStamina: avg, hasData: !avgVals.isEmpty)
        }
    }

    private var weekLabel: String {
        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "zh_CN")
        fmt.dateFormat = "M月d日"
        let start = fmt.string(from: weekStart)
        let endDate = Calendar.current.date(byAdding: .day, value: 6, to: weekStart) ?? weekStart
        let end = fmt.string(from: endDate)
        return "本周 · \(start) – \(end)"
    }

    private var avgStaminaThisWeek: Double {
        let vals = thisWeekSessions.compactMap(\.avgStamina)
        guard !vals.isEmpty else { return 0 }
        return vals.reduce(0, +) / Double(vals.count)
    }

    private var longestStreakMin: Int {
        Int((thisWeekSessions.map(\.duration).max() ?? 0) / 60)
    }

    // MARK: - Body

    var body: some View {
        NavigationStack {
            content
                .background(Color(UIColor.systemGroupedBackground).ignoresSafeArea())
                .navigationTitle("历史")
                .navigationBarTitleDisplayMode(.large)
                .searchable(text: $searchText, prompt: "搜索会话")
                .alert("导出失败", isPresented: $showExportError) {
                    Button("好") {}
                } message: {
                    Text(exportError ?? "未知错误")
                }
        }
    }

    @ViewBuilder
    private var content: some View {
        if completedSessions.isEmpty {
            emptyState
        } else {
            scrollableContent
        }
    }

    // MARK: - Scrollable Content

    private var scrollableContent: some View {
        ScrollView {
            LazyVStack(spacing: 20) {
                if !thisWeekSessions.isEmpty {
                    HistoryWeekHero(
                        bars: weekBars,
                        weekLabel: weekLabel,
                        totalSeconds: thisWeekSessions.reduce(0) { $0 + $1.duration },
                        sessionCount: thisWeekSessions.count,
                        avgStamina: avgStaminaThisWeek,
                        longestStreakMin: longestStreakMin
                    )
                }

                if trendPoints.contains(where: \.hasData) {
                    HistoryTrendCard(points: trendPoints)
                }

                ForEach(dayGroups, id: \.date) { group in
                    daySection(date: group.date, sessions: group.sessions)
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)
            .padding(.bottom, 24)
        }
    }

    // MARK: - Day Section

    private func daySection(date: Date, sessions: [Session]) -> some View {
        let total = sessions.reduce(0.0) { $0 + $1.duration }
        let cal = Calendar.current
        let isToday = cal.isDateInToday(date)
        let isYesterday = cal.isDateInYesterday(date)

        let title: String = {
            if isToday { return "今天" }
            if isYesterday { return "昨天" }
            let fmt = DateFormatter()
            fmt.locale = Locale(identifier: "zh_CN")
            fmt.dateFormat = "M月d日 EEEE"
            return fmt.string(from: date)
        }()

        return VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(title)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.primary)
                Spacer()
                Text(Flux.formatDuration(total))
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 4)
            .padding(.top, 4)

            VStack(spacing: 10) {
                ForEach(sessions) { session in
                    NavigationLink {
                        SessionDetailView(session: session)
                    } label: {
                        HistorySessionCard(session: session)
                    }
                    .buttonStyle(.plain)
                    .contextMenu {
                        Button {
                            exportSession(session)
                        } label: {
                            Label("导出", systemImage: "square.and.arrow.up")
                        }
                        Button(role: .destructive) {
                            modelContext.delete(session)
                            modelContext.saveLogged()
                        } label: {
                            Label("删除", systemImage: "trash")
                        }
                    }
                }
            }
        }
    }

    // MARK: - Empty

    private var emptyState: some View {
        ContentUnavailableView {
            Label("暂无记录", systemImage: "clock.arrow.circlepath")
        } description: {
            Text("开始录制后，记录将显示在这里")
        }
    }

    // MARK: - Actions

    private func exportSession(_ session: Session) {
        do {
            let url = try ExportManager.shareURL(for: session)
            FluxShare.shareFile(url: url)
        } catch {
            exportError = error.localizedDescription
            showExportError = true
        }
    }
}
