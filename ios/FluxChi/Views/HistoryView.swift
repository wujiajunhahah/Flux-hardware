import SwiftUI
import SwiftData

struct HistoryView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Session.startedAt, order: .reverse) private var sessions: [Session]

    @State private var searchText = ""
    @State private var shareURL: URL?
    @State private var showShareSheet = false

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

    var body: some View {
        NavigationStack {
            Group {
                if completedSessions.isEmpty {
                    emptyState
                } else {
                    sessionList
                }
            }
            .navigationTitle("历史")
            .searchable(text: $searchText, prompt: "搜索记录")
            .sheet(isPresented: $showShareSheet) {
                if let url = shareURL {
                    ShareSheet(items: [url])
                }
            }
        }
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("暂无记录", systemImage: "clock.arrow.circlepath")
        } description: {
            Text("开始录制后，历史记录将显示在这里")
        }
    }

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
            shareURL = try ExportManager.shareURL(for: session)
            showShareSheet = true
        } catch {}
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
