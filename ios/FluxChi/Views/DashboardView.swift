import SwiftUI
import SwiftData

struct DashboardView: View {
    @EnvironmentObject var service: FluxService
    @EnvironmentObject var bleManager: BLEManager
    @EnvironmentObject var sessionManager: SessionManager
    @EnvironmentObject var alertManager: AlertManager
    @EnvironmentObject var liveActivityManager: LiveActivityManager
    @Environment(\.modelContext) private var modelContext

    // 只查今日已完成的 session，避免加载全部历史
    @Query private var todaySessions: [Session]

    @State private var showSegmentPicker = false
    @State private var showFeedback = false
    @State private var finishedSession: Session?

    init() {
        let startOfDay = Calendar.current.startOfDay(for: Date())
        let predicate = #Predicate<Session> { session in
            session.startedAt >= startOfDay && session.endedAt != nil
        }
        _todaySessions = Query(filter: predicate, sort: \Session.startedAt, order: .reverse)
    }

    private var stamina: StaminaData? { service.state?.stamina }
    private var decision: DecisionData? { service.state?.decision }
    private var staminaValue: Double { stamina?.value ?? 0 }
    private var staminaState: StaminaState {
        StaminaState(rawValue: stamina?.state ?? "focused") ?? .focused
    }
    private var isLive: Bool { service.isConnected || bleManager.isConnected }

    // MARK: - Body

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    if !isLive { connectionBanner }

                    StaminaRingView(value: staminaValue, state: staminaState)
                        .drawingGroup() // GPU 离屏渲染，避免 AngularGradient 每帧重算
                        .padding(.vertical, 4)

                    if sessionManager.isRecording { recordingBar }

                    recommendationCard

                    // 提取为独立 struct，stamina 不变时不重绘
                    DimensionsRow(
                        consistency: stamina?.consistency ?? 0,
                        tension: stamina?.tension ?? 0,
                        fatigue: stamina?.fatigue ?? 0
                    )

                    TodaySummaryCard(
                        sessions: todaySessions,
                        decision: decision,
                        staminaState: staminaState,
                        suggestedBreakMin: stamina?.suggestedBreakMin ?? 0,
                        suggestedWorkMin: stamina?.suggestedWorkMin ?? 0
                    )

                    emgSection
                }
                .padding(.horizontal)
                .padding(.bottom, 80)
            }
            .navigationTitle("FocuX")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    FluxLiveIndicator(isLive: isLive)
                }
                ToolbarItem(placement: .topBarTrailing) { recordButton }
            }
            .onAppear {
                sessionManager.configure(
                    modelContext: modelContext,
                    stateProvider: { [weak service] in service?.state }
                )
            }
            .sheet(isPresented: $showFeedback) {
                if let s = finishedSession { FeedbackView(session: s) }
            }
        }
    }

    // MARK: - Connection

    @ViewBuilder
    private var connectionBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            Text("未连接")
                .font(.subheadline.weight(.medium))
            Spacer()
            if let err = service.connectionError {
                Text(err).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: 16))
    }

    // MARK: - Recommendation

    @ViewBuilder
    private var recommendationCard: some View {
        if let d = decision {
            let rec = Recommendation(rawValue: d.recommendation) ?? .keepWorking
            let urgencyColor = Flux.Colors.forUrgency(d.urgency)

            HStack(spacing: 14) {
                ZStack {
                    Circle()
                        .fill(urgencyColor.opacity(0.12))
                        .frame(width: 44, height: 44)
                    Image(systemName: rec.systemImage)
                        .font(.title3)
                        .foregroundStyle(urgencyColor)
                        .symbolRenderingMode(.hierarchical)
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text(rec.displayName)
                        .font(.subheadline.weight(.semibold))
                    if let reason = d.reasons.first {
                        Text(reason)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                }
                Spacer()
            }
            .padding(14)
            .background(.regularMaterial, in: .rect(cornerRadius: 16))
        }
    }

    // MARK: - EMG

    @ViewBuilder
    private var emgSection: some View {
        if let rms = service.state?.rms, !rms.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                FluxSectionLabel(title: "EMG", icon: "waveform")
                FluxEMGBars(rms: rms, height: 50)
                    .drawingGroup()
                    .padding(12)
                    .background(.ultraThinMaterial, in: .rect(cornerRadius: 16))
            }
        }
    }

    // MARK: - Recording

    private var recordButton: some View {
        Button {
            if sessionManager.isRecording {
                liveActivityManager.endActivity()
                if let session = sessionManager.endSession() {
                    let summary = SummaryEngine.generate(for: session)
                    SummaryEngine.apply(summary, to: session)
                    try? modelContext.save()
                    finishedSession = session
                    showFeedback = true
                }
            } else {
                let source: SessionSource = bleManager.isConnected ? .ble : .wifi
                sessionManager.startSession(source: source)

                let fmt = DateFormatter()
                fmt.locale = Locale(identifier: "zh_CN")
                fmt.dateFormat = "HH:mm"
                let title = "记录中 · \(fmt.string(from: Date()))"
                liveActivityManager.startActivity(title: title)
            }
        } label: {
            Image(systemName: sessionManager.isRecording ? "stop.circle.fill" : "record.circle")
                .font(.title3)
                .foregroundStyle(sessionManager.isRecording ? .red : (isLive ? .primary : .secondary))
                .symbolEffect(.pulse, isActive: sessionManager.isRecording)
        }
        .disabled(!isLive && !sessionManager.isRecording)
    }

    @ViewBuilder
    private var recordingBar: some View {
        HStack(spacing: 12) {
            Circle().fill(.red).frame(width: 8, height: 8)
                .opacity(sessionManager.isPaused ? 0.3 : 1)

            Text(Flux.formatDuration(sessionManager.elapsed))
                .font(.system(size: 15, weight: .semibold, design: .monospaced))
                .contentTransition(.numericText())

            Spacer()

            if let seg = sessionManager.activeSegment {
                Text(seg.label.displayName)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(seg.label.color)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(seg.label.color.opacity(0.12), in: Capsule())
            }

            Button {
                showSegmentPicker = true
            } label: {
                Image(systemName: "rectangle.split.1x2")
                    .font(.subheadline)
            }
            .confirmationDialog("新建分段", isPresented: $showSegmentPicker) {
                ForEach(SegmentLabel.allCases) { label in
                    Button(label.displayName) { sessionManager.addSegment(label: label) }
                }
            }

            Button {
                sessionManager.isPaused ? sessionManager.resumeSession() : sessionManager.pauseSession()
            } label: {
                Image(systemName: sessionManager.isPaused ? "play.fill" : "pause.fill")
                    .font(.subheadline)
            }
        }
        .padding(12)
        .background(.red.opacity(0.06), in: .rect(cornerRadius: 16))
    }
}

// MARK: - Dimensions Row (独立 struct，只在 value 变化时重绘)

private struct DimensionsRow: View, Equatable {
    let consistency: Double
    let tension: Double
    let fatigue: Double

    var body: some View {
        HStack(spacing: 12) {
            dimensionGauge("一致性", value: consistency, icon: "waveform.path", tint: .blue)
            dimensionGauge("紧张度", value: tension, icon: "arrow.up.right", tint: .orange)
            dimensionGauge("疲劳度", value: fatigue, icon: "flame", tint: .red)
        }
    }

    private func dimensionGauge(_ title: String, value: Double, icon: String, tint: Color) -> some View {
        VStack(spacing: 10) {
            Gauge(value: value) {
                Image(systemName: icon)
                    .font(.caption2)
            } currentValueLabel: {
                Text("\(Int(value * 100))")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
            }
            .gaugeStyle(.accessoryCircular)
            .tint(Gradient(colors: [tint.opacity(0.5), tint]))
            .scaleEffect(1.15)

            Text(title)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: 16))
    }
}

// MARK: - Today Summary Card (独立 struct，只在 sessions 变化时重绘)

private struct TodaySummaryCard: View {
    let sessions: [Session]
    let decision: DecisionData?
    let staminaState: StaminaState
    let suggestedBreakMin: Double
    let suggestedWorkMin: Double

    private var totalMinutes: Int {
        Int(sessions.reduce(0) { $0 + $1.duration } / 60)
    }

    private var avgStamina: Double {
        let vals = sessions.compactMap(\.avgStamina)
        guard !vals.isEmpty else { return 0 }
        return vals.reduce(0, +) / Double(vals.count)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("今日概览", systemImage: "chart.bar.fill")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.primary)

            HStack(spacing: 0) {
                statCell(
                    value: "\(sessions.count)",
                    label: "场次",
                    icon: "flame.fill",
                    tint: .orange
                )
                statCell(
                    value: totalMinutes > 0 ? "\(totalMinutes)m" : "—",
                    label: "总时长",
                    icon: "clock.fill",
                    tint: .blue
                )
                statCell(
                    value: avgStamina > 0 ? "\(Int(avgStamina))" : "—",
                    label: "平均续航",
                    icon: "heart.fill",
                    tint: .red
                )
            }

            if let d = decision {
                HStack(spacing: 16) {
                    progressBar(
                        current: d.continuousWorkMin,
                        label: "本次专注",
                        icon: "timer",
                        tint: .green
                    )
                    progressBar(
                        current: d.totalWorkMin,
                        label: "今日累计",
                        icon: "hourglass",
                        tint: .blue
                    )
                }
            }
        }
        .padding(16)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: 16))
    }

    private func statCell(value: String, label: String, icon: String, tint: Color) -> some View {
        VStack(spacing: 6) {
            ZStack {
                Circle()
                    .fill(tint.opacity(0.1))
                    .frame(width: 32, height: 32)
                Image(systemName: icon)
                    .font(.system(size: 13))
                    .foregroundStyle(tint)
            }

            Text(value)
                .font(.system(size: 18, weight: .bold, design: .rounded))

            Text(label)
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    private func progressBar(current: Double, label: String, icon: String, tint: Color) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundStyle(tint)
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(label)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(current < 1 ? "<1 min" : "\(Int(current)) min")
                        .font(.caption.weight(.semibold).monospacedDigit())
                }

                ProgressView(value: min(current / 60, 1))
                    .tint(tint)
            }
        }
        .frame(maxWidth: .infinity)
    }
}

#Preview {
    DashboardView()
        .environmentObject(FluxService())
        .environmentObject(BLEManager())
        .environmentObject(SessionManager())
        .environmentObject(AlertManager())
        .environmentObject(LiveActivityManager())
}
