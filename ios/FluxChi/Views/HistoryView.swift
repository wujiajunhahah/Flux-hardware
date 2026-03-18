import SwiftUI
import SwiftData

struct HistoryView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Session.startedAt, order: .reverse) private var sessions: [Session]

    @State private var viewMode: ViewMode = .list
    @State private var searchText = ""
    @State private var exportError: String?
    @State private var showExportError = false

    enum ViewMode: String, CaseIterable {
        case list, calendar

        var icon: String {
            switch self {
            case .list:     return "list.bullet"
            case .calendar: return "calendar"
            }
        }
    }

    private var completedSessions: [Session] {
        sessions.filter { !$0.isActive }
    }

    private var todaySessions: [Session] {
        let startOfDay = Calendar.current.startOfDay(for: Date())
        return completedSessions.filter { $0.startedAt >= startOfDay }
    }

    private var filtered: [Session] {
        guard !searchText.isEmpty else { return completedSessions }
        return completedSessions.filter {
            $0.title.localizedCaseInsensitiveContains(searchText)
        }
    }

    private var grouped: [(String, [Session])] {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "zh_CN")
        formatter.doesRelativeDateFormatting = true
        formatter.dateStyle = .medium

        let dict = Dictionary(grouping: filtered) { formatter.string(from: $0.startedAt) }
        return dict.sorted {
            ($0.value.first?.startedAt ?? .distantPast) > ($1.value.first?.startedAt ?? .distantPast)
        }
    }

    var body: some View {
        NavigationStack {
            Group {
                if completedSessions.isEmpty {
                    emptyState
                } else {
                    switch viewMode {
                    case .list:
                        sessionList
                    case .calendar:
                        FluxCalendarView()
                    }
                }
            }
            .navigationTitle("历史")
            .searchable(text: $searchText, prompt: "搜索")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Picker("视图", selection: $viewMode) {
                            ForEach(ViewMode.allCases, id: \.self) { mode in
                                Label(mode == .list ? "列表" : "日历",
                                      systemImage: mode.icon)
                                .tag(mode)
                            }
                        }
                    } label: {
                        Image(systemName: viewMode.icon)
                            .font(.body)
                    }
                }
            }
            .alert("导出失败", isPresented: $showExportError) {
                Button("好") {}
            } message: {
                Text(exportError ?? "未知错误")
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

    // MARK: - List

    private var sessionList: some View {
        List {
            if !todaySessions.isEmpty {
                Section {
                    historyTodayOverview
                } header: {
                    Text("今日")
                }
            }

            ForEach(grouped, id: \.0) { dateStr, daySessions in
                Section {
                    ForEach(daySessions) { session in
                        NavigationLink(destination: SessionDetailView(session: session)) {
                            SessionRow(session: session)
                        }
                        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                            Button(role: .destructive) {
                                modelContext.delete(session)
                                try? modelContext.save()
                            } label: {
                                Label("删除", systemImage: "trash")
                            }

                            Button {
                                exportSession(session)
                            } label: {
                                Label("导出", systemImage: "square.and.arrow.up")
                            }
                            .tint(.blue)
                        }
                    }
                } header: {
                    Text(dateStr)
                }
            }
        }
        .listStyle(.insetGrouped)
    }

    // MARK: - History Today Overview (BIOSORA-style metric row)

    private var historyTodayOverview: some View {
        let totalMin = Int(todaySessions.reduce(0) { $0 + $1.duration } / 60)
        let avgVals = todaySessions.compactMap(\.avgStamina)
        let avgStamina = avgVals.isEmpty ? 0.0 : avgVals.reduce(0, +) / Double(avgVals.count)
        let pending = todaySessions.filter { $0.feedback == nil }.count

        return HStack(spacing: 16) {
            FluxMetricCard(title: "场次", value: "\(todaySessions.count)", icon: "number", compact: true)
            FluxMetricCard(title: "时长", value: totalMin > 0 ? "\(totalMin)m" : "—", icon: "clock", compact: true)
            FluxMetricCard(title: "续航", value: avgStamina > 0 ? "\(Int(avgStamina))" : "—", icon: "bolt.fill", tint: Flux.Colors.forStaminaValue(avgStamina), compact: true)
            if pending > 0 {
                Spacer()
                Text("\(pending) 待反馈")
                    .font(.caption2)
                    .foregroundStyle(Flux.Colors.warning)
            }
        }
    }

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

// MARK: - Session Row (BIOSORA-inspired: 圆环 + 内容 + sparkline + 状态)

private struct SessionRow: View {
    let session: Session

    private var timeRange: String {
        let fmt = DateFormatter()
        fmt.dateFormat = "HH:mm"
        let start = fmt.string(from: session.startedAt)
        let end = fmt.string(from: session.startedAt.addingTimeInterval(session.duration))
        return "\(start)–\(end)"
    }

    var body: some View {
        HStack(spacing: 12) {
            // 续航圆环指示器 — BIOSORA 式圆形数据呈现
            if let avg = session.avgStamina {
                staminaIndicator(avg)
            }

            // 内容区
            VStack(alignment: .leading, spacing: 3) {
                // 第一层：时间范围
                Text(timeRange)
                    .font(.subheadline.weight(.medium))

                // 第二层：时长 + 分段
                HStack(spacing: 8) {
                    Text(Flux.formatDuration(session.duration))
                        .foregroundStyle(.secondary)
                    if let count = session.segmentCount, count > 0 {
                        Text("\(count) 段")
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.caption)

                // 第三层：迷你 Sparkline — BIOSORA 式数据曲线
                if let curve = session.staminaCurveData,
                   let values = try? JSONDecoder().decode([Double].self, from: curve),
                   values.count >= 2 {
                    miniSparkline(values)
                }
            }

            Spacer()

            // 右侧状态区
            VStack(alignment: .trailing, spacing: 4) {
                // 待反馈指示 — 暖琥珀圆点（替代原纯红点）
                if session.feedback == nil {
                    HStack(spacing: 3) {
                        Circle()
                            .fill(Flux.Colors.warning)
                            .frame(width: 6, height: 6)
                        Text("待反馈")
                            .font(.system(size: 9))
                            .foregroundStyle(Flux.Colors.warning)
                    }
                }

                // 来源图标
                Image(systemName: session.source.icon)
                    .font(.caption)
                    .foregroundStyle(.quaternary)
            }
        }
        .padding(.vertical, 2)
    }

    // MARK: - Stamina Indicator (与 CalendarSessionCard 统一风格)

    private func staminaIndicator(_ avg: Double) -> some View {
        let color = Flux.Colors.forStaminaValue(avg)

        return ZStack {
            Circle()
                .stroke(color.opacity(0.12), lineWidth: 3)
                .frame(width: 38, height: 38)
            Circle()
                .trim(from: 0, to: avg / 100)
                .stroke(color, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .frame(width: 38, height: 38)
            Text("\(Int(avg))")
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .foregroundStyle(color)
        }
    }

    // MARK: - Mini Sparkline (BIOSORA 式迷你数据曲线)

    private func miniSparkline(_ values: [Double]) -> some View {
        let step = max(1, values.count / 30)
        let sampled = stride(from: 0, to: values.count, by: step).map { values[$0] }

        return GeometryReader { geo in
            let maxV = sampled.max() ?? 100
            let minV = sampled.min() ?? 0
            let range = max(maxV - minV, 1)

            Path { path in
                for (i, v) in sampled.enumerated() {
                    let x = geo.size.width * CGFloat(i) / CGFloat(max(sampled.count - 1, 1))
                    let y = geo.size.height * (1 - CGFloat((v - minV) / range))
                    if i == 0 { path.move(to: CGPoint(x: x, y: y)) }
                    else { path.addLine(to: CGPoint(x: x, y: y)) }
                }
            }
            .stroke(Flux.Colors.accent.opacity(0.5), lineWidth: 1.5)
        }
        .frame(height: 14)
    }
}
