import SwiftUI
import SwiftData
import Charts

struct SessionDetailView: View {
    let session: Session
    @Environment(\.modelContext) private var modelContext
    @State private var showFeedback = false
    @State private var exportError: String?
    @State private var showExportError = false
    @State private var nlpSummary: String?
    @State private var isGeneratingSummary = false

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

    private var avgConsistency: Double {
        let snaps = allSnapshots
        guard !snaps.isEmpty else { return 0 }
        return snaps.map(\.consistency).reduce(0, +) / Double(snaps.count)
    }

    private var avgTension: Double {
        let snaps = allSnapshots
        guard !snaps.isEmpty else { return 0 }
        return snaps.map(\.tension).reduce(0, +) / Double(snaps.count)
    }

    private var avgFatigue: Double {
        let snaps = allSnapshots
        guard !snaps.isEmpty else { return 0 }
        return snaps.map(\.fatigue).reduce(0, +) / Double(snaps.count)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: Flux.Spacing.section) {
                summaryCard
                StaminaCurveChart(snapshots: chartSnapshots)

                HStack(alignment: .top, spacing: Flux.Spacing.item) {
                    DimensionRadarView(
                        consistency: avgConsistency,
                        tension: avgTension,
                        fatigue: avgFatigue
                    )
                    .frame(maxWidth: .infinity)
                }

                FocusClockView(snapshots: allSnapshots, sessionStart: session.startedAt)
                ADHDDotMap(snapshots: allSnapshots, sessionStart: session.startedAt)
                statsGrid
                stateTimeline
                segmentsList
                feedbackSection
            }
            .padding()
        }
        .navigationTitle(session.title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar(.hidden, for: .tabBar)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button { exportSession() } label: {
                        Label("导出 JSON", systemImage: "square.and.arrow.up")
                    }
                    Button { generateNLPSummary() } label: {
                        Label("AI 总结", systemImage: "sparkles")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .sheet(isPresented: $showFeedback) {
            FeedbackView(session: session)
        }
        .alert("导出失败", isPresented: $showExportError) {
            Button("好") {}
        } message: {
            Text(exportError ?? "未知错误")
        }
        .task {
            if nlpSummary == nil && session.summaryText == nil {
                generateNLPSummary()
            }
        }
    }

    // MARK: - NLP Summary Card

    @ViewBuilder
    private var summaryCard: some View {
        let text = nlpSummary ?? session.summaryText
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                FluxSectionLabel(title: "智能总结", icon: "sparkles")
                Spacer()
                if isGeneratingSummary {
                    ProgressView()
                        .scaleEffect(0.7)
                }
            }

            if let text, !text.isEmpty {
                Text(text)
                    .font(.subheadline)
                    .lineSpacing(6)
                    .foregroundStyle(.primary.opacity(0.85))
            } else if !isGeneratingSummary {
                Text(session.summaryText ?? "点击右上角菜单生成 AI 总结")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.large))
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
                                Text(label.displayName).font(.caption2).foregroundStyle(.secondary)
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
                tint: Flux.Colors.success
            )
            FluxMetricCard(
                title: "最低",
                value: "\(Int(session.minStamina ?? 0))",
                icon: "arrow.down",
                tint: Flux.Colors.accent
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

    // MARK: - Actions

    private func exportSession() {
        do {
            let url = try ExportManager.shareURL(for: session)
            FluxShare.shareFile(url: url)
        } catch {
            exportError = error.localizedDescription
            showExportError = true
        }
    }

    private func generateNLPSummary() {
        guard !isGeneratingSummary else { return }
        isGeneratingSummary = true

        Task {
            if #available(iOS 26.0, *) {
                let engine = NLPSummaryEngine.shared
                let result = await engine.generateSummary(for: session)
                nlpSummary = result
                session.summaryText = result
                modelContext.saveLogged()
            } else {
                let summary = SummaryEngine.generate(for: session)
                SummaryEngine.apply(summary, to: session)
                nlpSummary = summary.text
                modelContext.saveLogged()
            }
            isGeneratingSummary = false
        }
    }
}
