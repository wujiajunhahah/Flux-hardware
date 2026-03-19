import SwiftUI

/// 日志查看器视图
struct LogViewerView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var logger = FluxLogger.shared

    @State private var selectedCategory: FluxLogCategory?
    @State private var selectedLevel: FluxLogLevel?
    @State private var searchText = ""
    @State private var isSearching = false

    private var filteredEntries: [FluxLogEntry] {
        var result = logger.entries

        if let category = selectedCategory {
            result = result.filter { $0.category == category }
        }

        if let level = selectedLevel {
            result = result.filter { $0.level >= level }
        }

        if !searchText.isEmpty {
            result = result.filter {
                $0.message.localizedCaseInsensitiveContains(searchText) ||
                $0.errorDescription?.localizedCaseInsensitiveContains(searchText) == true
            }
        }

        return result.reversed()
    }

    var body: some View {
        NavigationStack {
            Group {
                if filteredEntries.isEmpty {
                    emptyStateView
                } else {
                    entriesList
                }
            }
            .navigationTitle("日志查看器")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("完成") { dismiss() }
                }
                ToolbarItem(placement: .primaryAction) {
                    Menu {
                        filterMenu
                    } label: {
                        Image(systemName: "line.3.horizontal.decrease.circle")
                    }
                }
            }
            .searchable(text: $searchText, prompt: "搜索日志")
        }
    }

    // MARK: - Empty State

    private var emptyStateView: some View {
        ContentUnavailableView {
            Label("无日志", systemImage: "doc.text")
        } description: {
            VStack(spacing: 8) {
                if !selectedCategory.isNil || !selectedLevel.isNil {
                    Text("当前筛选条件下没有日志")
                    Button("清除筛选") {
                        selectedCategory = nil
                        selectedLevel = nil
                    }
                    .buttonStyle(.bordered)
                } else if logger.entries.isEmpty {
                    Text("暂无日志记录")
                }
            }
        }
    }

    // MARK: - Entries List

    private var entriesList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 1) {
                ForEach(filteredEntries) { entry in
                    LogEntryCell(entry: entry)
                        .onTapGesture {
                            // TODO: 显示详情
                        }
                }
            }
            .padding(.vertical)
        }
        .background(Color(uiColor: .systemGroupedBackground))
    }

    // MARK: - Filter Menu

    @ViewBuilder
    private var filterMenu: some View {
        Button {
            selectedCategory = nil
            selectedLevel = nil
        } label: {
            Label("全部", systemImage: "line.3.horizontal.decrease.circle")
        }

        Divider()

        Menu("分类") {
            Button("全部") { selectedCategory = nil }
            Divider()
            ForEach(FluxLogCategory.allCases, id: \.self) { category in
                Button {
                    selectedCategory = category
                } label: {
                    HStack {
                        Image(systemName: category.icon)
                        Text(category.rawValue)
                    }
                }
            }
        }

        Menu("级别") {
            Button("全部") { selectedLevel = nil }
            Divider()
            ForEach(FluxLogLevel.allCases, id: \.self) { level in
                Button {
                    selectedLevel = level
                } label: {
                    HStack {
                        Image(systemName: level.icon)
                        Text(level.label)
                    }
                }
            }
        }
    }
}

// MARK: - Log Entry Cell

private struct LogEntryCell: View {
    let entry: FluxLogEntry

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // Level icon
            Image(systemName: entry.level.icon)
                .font(.caption)
                .foregroundStyle(entry.level.color)
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 4) {
                // Header
                HStack(spacing: 8) {
                    Text(entry.category.rawValue)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(entry.category.color)

                    Spacer()

                    Text(entry.timestamp, style: .time)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                // Message
                Text(entry.message)
                    .font(.caption)
                    .foregroundStyle(.primary)
                    .fixedSize(horizontal: false, vertical: true)

                // Error description (if any)
                if let error = entry.errorDescription {
                    Text(error)
                        .font(.caption2)
                        .foregroundStyle(.red)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .padding(.horizontal)
    }
}

// Helper for optional comparison
extension Optional {
    var isNil: Bool {
        self == nil
    }
}

#Preview {
    LogViewerView()
}
