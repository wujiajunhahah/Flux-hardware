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

// MARK: - Session Row

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
            // Stamina gauge
            if let avg = session.avgStamina {
                staminaIndicator(avg)
            }

            // Content
            VStack(alignment: .leading, spacing: 3) {
                Text(timeRange)
                    .font(.subheadline.weight(.medium))

                HStack(spacing: 8) {
                    Text(Flux.formatDuration(session.duration))
                        .foregroundStyle(.secondary)

                    if let count = session.segmentCount, count > 0 {
                        Text("\(count) 段")
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.caption)
            }

            Spacer()

            // Source icon
            Image(systemName: session.source.icon)
                .font(.caption)
                .foregroundStyle(.quaternary)
        }
        .padding(.vertical, Flux.Spacing.tight)
    }

    private func staminaIndicator(_ avg: Double) -> some View {
        let color: Color = avg > 60 ? .green : avg > 30 ? .orange : .red

        return ZStack {
            Circle()
                .stroke(color.opacity(Flux.Opacity.semiStrong), lineWidth: Flux.Sizes.strokeThick)
                .frame(width: 36, height: 36)
            Circle()
                .trim(from: 0, to: avg / 100)
                .stroke(color, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .frame(width: 36, height: 36)

            Text("\(Int(avg))")
                .font(.system(size: Flux.Sizes.textLabel + 2, weight: .bold, design: .rounded))
                .foregroundStyle(color)
        }
    }
}
