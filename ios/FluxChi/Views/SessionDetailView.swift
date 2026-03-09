import SwiftUI
import SwiftData
import Charts

struct SessionDetailView: View {
    let session: Session
    @Environment(\.modelContext) private var modelContext
    @State private var showFeedback = false
    @State private var showExportSheet = false
    @State private var exportURL: URL?

    private var allSnapshots: [FluxSnapshot] {
        session.segments
            .flatMap { $0.snapshots }
            .sorted { $0.timestamp < $1.timestamp }
    }

    private var chartSnapshots: [FluxSnapshot] {
        let all = allSnapshots
        guard all.count > 200 else { return all }
        let step = Double(all.count) / 200.0
        return (0..<200).map { all[min(Int(Double($0) * step), all.count - 1)] }
    }

    var body: some View {
        ScrollView {
            VStack(spacing: Flux.Spacing.section) {
                summaryCard
                staminaChart
                stateTimeline
                statsGrid
                segmentsList
                feedbackSection
            }
            .padding()
        }
        .navigationTitle(session.title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button { exportSession() } label: {
                        Label("导出 JSON", systemImage: "square.and.arrow.up")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .sheet(isPresented: $showFeedback) {
            FeedbackView(session: session)
        }
        .sheet(isPresented: $showExportSheet) {
            if let url = exportURL { ShareSheet(items: [url]) }
        }
    }

    // MARK: - Summary

    @ViewBuilder
    private var summaryCard: some View {
        if let text = session.summaryText, !text.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                FluxSectionLabel(title: "总结", icon: "text.quote")
                Text(text)
                    .font(.subheadline)
                    .lineSpacing(4)
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.large))
        }
    }

    // MARK: - Chart

    @ViewBuilder
    private var staminaChart: some View {
        if !chartSnapshots.isEmpty {
            VStack(alignment: .leading, spacing: Flux.Spacing.item) {
                FluxSectionLabel(title: "Stamina 曲线", icon: "chart.xyaxis.line")

                Chart {
                    ForEach(Array(chartSnapshots.enumerated()), id: \.offset) { _, snap in
                        LineMark(
                            x: .value("时间", snap.timestamp),
                            y: .value("Stamina", snap.stamina)
                        )
                        .interpolationMethod(.catmullRom)
                    }

                    RuleMark(y: .value("高效", 60))
                        .foregroundStyle(.green.opacity(0.3))
                        .lineStyle(StrokeStyle(dash: [4]))

                    RuleMark(y: .value("警告", 30))
                        .foregroundStyle(.red.opacity(0.3))
                        .lineStyle(StrokeStyle(dash: [4]))

                    ForEach(Array(session.segments.dropFirst().enumerated()), id: \.offset) { _, seg in
                        RuleMark(x: .value("分段", seg.startedAt))
                            .foregroundStyle(.secondary.opacity(0.4))
                            .lineStyle(StrokeStyle(dash: [2, 4]))
                    }
                }
                .chartYScale(domain: 0...100)
                .chartYAxis {
                    AxisMarks(values: [0, 30, 60, 100])
                }
                .chartForegroundStyleScale([
                    "Stamina": Flux.Colors.accent
                ])
                .frame(height: 200)
                .padding()
                .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))
            }
        }
    }

    // MARK: - State Timeline

    @ViewBuilder
    private var stateTimeline: some View {
        if !session.segments.isEmpty {
            VStack(alignment: .leading, spacing: Flux.Spacing.item) {
                FluxSectionLabel(title: "状态分布", icon: "chart.bar.fill")

                GeometryReader { geo in
                    let totalDuration = max(session.duration, 1)
                    HStack(spacing: 1) {
                        ForEach(session.segments) { seg in
                            let fraction = seg.duration / totalDuration
                            RoundedRectangle(cornerRadius: 3)
                                .fill(seg.label.color.gradient)
                                .frame(width: max(2, geo.size.width * fraction))
                                .overlay {
                                    if fraction > 0.15 {
                                        Text(seg.label.displayName)
                                            .font(.system(size: 9, weight: .medium))
                                            .foregroundStyle(.white)
                                    }
                                }
                        }
                    }
                }
                .frame(height: 28)
                .clipShape(.rect(cornerRadius: 6))

                HStack(spacing: 12) {
                    ForEach(SegmentLabel.allCases) { label in
                        let count = session.segments.filter { $0.label == label }.count
                        if count > 0 {
                            HStack(spacing: 4) {
                                Circle().fill(label.color).frame(width: 8, height: 8)
                                Text(label.displayName)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - Stats

    private var statsGrid: some View {
        LazyVGrid(columns: [.init(), .init(), .init()], spacing: Flux.Spacing.item) {
            FluxMetricCard(
                title: "时长",
                value: Flux.formatMinutes(session.duration / 60),
                icon: "clock.fill",
                tint: .blue
            )
            FluxMetricCard(
                title: "平均",
                value: "\(Int(session.avgStamina ?? 0))",
                icon: "gauge.with.dots.needle.67percent",
                tint: .green
            )
            FluxMetricCard(
                title: "最低",
                value: "\(Int(session.minStamina ?? 0))",
                icon: "arrow.down",
                tint: .red
            )
        }
    }

    // MARK: - Segments List

    private var segmentsList: some View {
        VStack(alignment: .leading, spacing: Flux.Spacing.item) {
            FluxSectionLabel(title: "分段", icon: "rectangle.split.1x2")

            ForEach(session.segments) { seg in
                HStack {
                    Label(seg.label.displayName, systemImage: seg.label.icon)
                        .font(.subheadline)
                        .foregroundStyle(seg.label.color)

                    Spacer()

                    Text(Flux.formatDuration(seg.duration))
                        .font(Flux.Typography.mono)
                        .foregroundStyle(.secondary)

                    Text("\(seg.snapshots.count) 帧")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                .padding(.vertical, 6)
            }
        }
    }

    // MARK: - Feedback

    private var feedbackSection: some View {
        Group {
            if let fb = session.feedback {
                VStack(alignment: .leading, spacing: Flux.Spacing.item) {
                    FluxSectionLabel(title: "反馈", icon: "hand.thumbsup.fill")

                    HStack {
                        Label(fb.feeling.displayName, systemImage: fb.feeling.icon)
                            .foregroundStyle(fb.feeling.color)
                        Spacer()
                        HStack(spacing: 2) {
                            ForEach(1...5, id: \.self) { i in
                                Image(systemName: i <= fb.accuracyRating ? "star.fill" : "star")
                                    .font(.caption)
                                    .foregroundStyle(i <= fb.accuracyRating ? .yellow : .secondary)
                            }
                        }
                    }

                    if !fb.notes.isEmpty {
                        Text(fb.notes)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding()
                .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))
            } else {
                Button { showFeedback = true } label: {
                    Label("添加反馈", systemImage: "plus.bubble")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))
                }
            }
        }
    }

    private func exportSession() {
        do {
            exportURL = try ExportManager.shareURL(for: session)
            showExportSheet = true
        } catch {}
    }
}
