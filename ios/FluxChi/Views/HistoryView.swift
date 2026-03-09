import SwiftUI
import SwiftData

struct HistoryView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Session.startedAt, order: .reverse) private var sessions: [Session]

    @State private var viewMode: ViewMode = .calendar
    @State private var searchText = ""
    @State private var exportError: String?
    @State private var showExportError = false

    enum ViewMode: String, CaseIterable {
        case calendar, list

        var icon: String {
            switch self {
            case .calendar: return "calendar"
            case .list:     return "list.bullet"
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
        formatter.dateFormat = "yyyy年M月d日"

        let dict = Dictionary(grouping: filtered) { formatter.string(from: $0.startedAt) }
        return dict.sorted { ($0.value.first?.startedAt ?? .distantPast) > ($1.value.first?.startedAt ?? .distantPast) }
    }

    // MARK: - Summary Stats

    private var todaySessions: [Session] {
        completedSessions.filter { Calendar.current.isDateInToday($0.startedAt) }
    }

    private var todayWorkMinutes: Int {
        Int(todaySessions.reduce(0.0) { $0 + ($1.workDurationSec ?? 0) } / 60)
    }

    private var todayAvgStamina: Double {
        let vals = todaySessions.compactMap(\.avgStamina)
        guard !vals.isEmpty else { return 0 }
        return vals.reduce(0, +) / Double(vals.count)
    }

    private var weekSessionCount: Int {
        let weekAgo = Calendar.current.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        return completedSessions.filter { $0.startedAt >= weekAgo }.count
    }

    var body: some View {
        NavigationStack {
            Group {
                if completedSessions.isEmpty {
                    emptyState
                } else {
                    VStack(spacing: 0) {
                        todaySummaryBar
                        Divider()
                        viewToggle
                        switch viewMode {
                        case .calendar:
                            FluxCalendarView()
                        case .list:
                            sessionList
                        }
                    }
                }
            }
            .navigationTitle("历史")
            .searchable(text: $searchText, prompt: "搜索记录")
            .alert("导出失败", isPresented: $showExportError) {
                Button("好") {}
            } message: {
                Text(exportError ?? "未知错误")
            }
        }
    }

    // MARK: - Today Summary

    private var todaySummaryBar: some View {
        HStack(spacing: 20) {
            VStack(spacing: 2) {
                Text("\(todaySessions.count)")
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                Text("今日记录")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Divider().frame(height: 30)

            VStack(spacing: 2) {
                Text("\(todayWorkMinutes)")
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                    .foregroundStyle(.blue)
                Text("工作分钟")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Divider().frame(height: 30)

            VStack(spacing: 2) {
                Text("\(Int(todayAvgStamina))")
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                    .foregroundStyle(todayAvgStamina >= 60 ? .green : .orange)
                Text("平均专注")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Divider().frame(height: 30)

            VStack(spacing: 2) {
                Text("\(weekSessionCount)")
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                    .foregroundStyle(.purple)
                Text("本周")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
    }

    // MARK: - View Toggle

    private var viewToggle: some View {
        Picker("视图", selection: $viewMode) {
            ForEach(ViewMode.allCases, id: \.self) { mode in
                Image(systemName: mode.icon).tag(mode)
            }
        }
        .pickerStyle(.segmented)
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    // MARK: - Empty

    private var emptyState: some View {
        ContentUnavailableView {
            Label("暂无记录", systemImage: "clock.arrow.circlepath")
        } description: {
            Text("开始录制后，历史记录将显示在这里")
        }
    }

    // MARK: - List

    private var sessionList: some View {
        List {
            ForEach(grouped, id: \.0) { dateStr, daySessions in
                Section(dateStr) {
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
                }
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

// MARK: - Session Row

private struct SessionRow: View {
    let session: Session

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(session.title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .lineLimit(1)

                Spacer()

                Label(session.source.displayName, systemImage: session.source.icon)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 16) {
                Label(Flux.formatDuration(session.duration), systemImage: "clock")

                if let avg = session.avgStamina {
                    Label("\(Int(avg))", systemImage: "gauge.with.dots.needle.67percent")
                        .foregroundStyle(avg > 60 ? .green : avg > 30 ? .orange : .red)
                }

                if let count = session.segmentCount, count > 0 {
                    Label("\(count) 段", systemImage: "rectangle.split.1x2")
                }

                if session.feedback != nil {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.caption)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}
