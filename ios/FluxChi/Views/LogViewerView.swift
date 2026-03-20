import SwiftUI

/// 日志查看器视图
struct LogViewerView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var logger = FluxLogger.shared

    @State private var selectedCategory: FluxLogCategory?
    @State private var selectedLevel: FluxLogLevel?
    @State private var searchText = ""
    @State private var selectedEntry: FluxLogEntry?

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
                $0.errorDescription?.localizedCaseInsensitiveContains(searchText) == true ||
                $0.fileName.localizedCaseInsensitiveContains(searchText) ||
                $0.function.localizedCaseInsensitiveContains(searchText)
            }
        }

        return result.reversed()
    }

    private var activeFilterCount: Int {
        (selectedCategory != nil ? 1 : 0) + (selectedLevel != nil ? 1 : 0)
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
                        Image(systemName: activeFilterCount > 0
                              ? "line.3.horizontal.decrease.circle.fill"
                              : "line.3.horizontal.decrease.circle")
                    }
                }
            }
            .searchable(text: $searchText, prompt: "搜索日志 / 文件名 / 函数")
            .sheet(item: $selectedEntry) { entry in
                LogDetailSheet(entry: entry)
            }
        }
    }

    // MARK: - Empty State

    private var emptyStateView: some View {
        ContentUnavailableView {
            Label("无日志", systemImage: "doc.text")
        } description: {
            VStack(spacing: 8) {
                if selectedCategory != nil || selectedLevel != nil {
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
                            selectedEntry = entry
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
                        if selectedCategory == category {
                            Image(systemName: "checkmark")
                        }
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
                        if selectedLevel == level {
                            Image(systemName: "checkmark")
                        }
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
                // Header: category + source + time
                HStack(spacing: 6) {
                    Text(entry.category.rawValue)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(entry.category.color)

                    if !entry.sourceLocation.isEmpty {
                        Text(entry.sourceLocation)
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.tertiary)
                    }

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
                    .lineLimit(3)

                // Error description
                if let error = entry.errorDescription {
                    Text(error)
                        .font(.caption2)
                        .foregroundStyle(.red)
                        .fixedSize(horizontal: false, vertical: true)
                        .lineLimit(2)
                }
            }

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .padding(.horizontal)
    }
}

// MARK: - Log Detail Sheet

private struct LogDetailSheet: View {
    let entry: FluxLogEntry
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Flux.Spacing.group) {
                    // Header
                    HStack(spacing: 8) {
                        Image(systemName: entry.level.icon)
                            .foregroundStyle(entry.level.color)
                        Text(entry.level.label)
                            .font(.headline)
                        Text(entry.category.rawValue)
                            .font(.subheadline)
                            .foregroundStyle(entry.category.color)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 2)
                            .background(entry.category.color.opacity(0.1), in: Capsule())
                    }

                    // Timestamp
                    LabeledContent("时间") {
                        Text(entry.timestamp, format: .dateTime.hour().minute().second())
                            .font(.system(.body, design: .monospaced))
                    }

                    // Source location
                    if !entry.sourceLocation.isEmpty {
                        LabeledContent("位置") {
                            Text(entry.sourceLocation)
                                .font(.system(.body, design: .monospaced))
                                .foregroundStyle(.blue)
                        }
                        LabeledContent("函数") {
                            Text(entry.function)
                                .font(.system(.caption, design: .monospaced))
                                .lineLimit(2)
                        }
                    }

                    Divider()

                    // Message
                    VStack(alignment: .leading, spacing: 4) {
                        Text("消息")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Text(entry.message)
                            .font(.body)
                            .textSelection(.enabled)
                    }

                    // Error
                    if let error = entry.errorDescription {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("错误")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.red)
                            Text(error)
                                .font(.body)
                                .foregroundStyle(.red.opacity(0.8))
                                .textSelection(.enabled)
                        }
                    }

                    Divider()

                    // Copy button
                    Button {
                        UIPasteboard.general.string = entry.formatDetailed()
                    } label: {
                        Label("复制完整日志", systemImage: "doc.on.doc")
                    }
                    .buttonStyle(.bordered)
                }
                .padding()
            }
            .navigationTitle("日志详情")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("关闭") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }
}

#Preview {
    LogViewerView()
}
