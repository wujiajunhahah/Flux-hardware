import SwiftUI
import SwiftData

struct DashboardView: View {
    @EnvironmentObject var service: FluxService
    @EnvironmentObject var bleManager: BLEManager
    @EnvironmentObject var sessionManager: SessionManager
    @EnvironmentObject var alertManager: AlertManager
    @EnvironmentObject var liveActivityManager: LiveActivityManager
    @Environment(\.modelContext) private var modelContext

    @State private var showSegmentPicker = false
    @State private var showFeedback = false
    @State private var finishedSession: Session?

    private var stamina: StaminaData? { service.state?.stamina }
    private var decision: DecisionData? { service.state?.decision }
    private var staminaValue: Double { stamina?.value ?? 0 }
    private var staminaState: StaminaState {
        StaminaState(rawValue: stamina?.state ?? "focused") ?? .focused
    }
    private var isLive: Bool { service.isConnected || bleManager.isConnected }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    if !isLive { connectionBanner }
                    staminaSection
                    if sessionManager.isRecording { recordingBar }
                    recommendationCard
                    dimensionsRow
                    statsRow
                    emgSection
                }
                .padding(.horizontal)
                .padding(.bottom, 80)
            }
            .navigationTitle("FluxChi")
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

    // MARK: - Stamina Ring

    private var staminaSection: some View {
        StaminaRingView(value: staminaValue, state: staminaState)
            .padding(.vertical, 4)
    }

    // MARK: - Recommendation

    @ViewBuilder
    private var recommendationCard: some View {
        if let d = decision {
            let rec = Recommendation(rawValue: d.recommendation) ?? .keepWorking
            HStack(spacing: 12) {
                Image(systemName: rec.systemImage)
                    .font(.title2)
                    .foregroundStyle(Flux.Colors.forUrgency(d.urgency))
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 40)

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

    // MARK: - Dimensions (3 pill cards with animated progress)

    private var dimensionsRow: some View {
        HStack(spacing: 10) {
            dimensionCard(
                "一致性",
                value: stamina?.consistency ?? 0,
                icon: "waveform.path",
                tint: .blue,
                description: "信号稳定度"
            )
            dimensionCard(
                "紧张度",
                value: stamina?.tension ?? 0,
                icon: "arrow.up.right",
                tint: .orange,
                description: "肌肉张力"
            )
            dimensionCard(
                "疲劳度",
                value: stamina?.fatigue ?? 0,
                icon: "flame",
                tint: .red,
                description: "频谱衰减"
            )
        }
    }

    private func dimensionCard(_ title: String, value: Double, icon: String,
                               tint: Color, description: String) -> some View {
        VStack(spacing: 8) {
            ZStack {
                Circle()
                    .stroke(tint.opacity(0.15), lineWidth: 4)
                    .frame(width: 44, height: 44)
                Circle()
                    .trim(from: 0, to: max(0.02, value))
                    .stroke(tint.gradient, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .frame(width: 44, height: 44)
                    .animation(.easeOut(duration: 0.3), value: value)

                Text("\(Int(value * 100))")
                    .font(.system(size: 13, weight: .bold, design: .rounded))
                    .contentTransition(.numericText(value: value))
            }

            Text(title)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)

            Text(description)
                .font(.system(size: 8))
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
        .background(.regularMaterial, in: .rect(cornerRadius: 16))
    }

    // MARK: - Stats

    private var statsRow: some View {
        HStack(spacing: 0) {
            compactStat(
                formatMin(decision?.continuousWorkMin ?? 0),
                "本次", "timer"
            )
            Divider().frame(height: 32)
            compactStat(
                formatMin(decision?.totalWorkMin ?? 0),
                "累计", "clock.fill"
            )
            Divider().frame(height: 32)
            compactStat(
                formatMin(staminaState == .recovering
                    ? (stamina?.suggestedBreakMin ?? 0)
                    : (stamina?.suggestedWorkMin ?? 0)),
                staminaState == .recovering ? "休息" : "专注",
                staminaState == .recovering ? "moon.fill" : "bolt.fill"
            )
        }
        .padding(.vertical, 10)
        .background(.regularMaterial, in: .rect(cornerRadius: 16))
    }

    private func compactStat(_ value: String, _ label: String, _ icon: String) -> some View {
        VStack(spacing: 3) {
            Image(systemName: icon).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.title3.weight(.bold).monospacedDigit())
            Text(label).font(.system(size: 9)).foregroundStyle(.secondary).tracking(0.5)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - EMG

    @ViewBuilder
    private var emgSection: some View {
        if let rms = service.state?.rms, !rms.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                FluxSectionLabel(title: "EMG", icon: "waveform")
                FluxEMGBars(rms: rms, height: 50)
                    .padding(12)
                    .background(.regularMaterial, in: .rect(cornerRadius: 16))
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

    // MARK: - Helpers

    private func formatMin(_ m: Double) -> String {
        m < 1 ? "<1" : "\(Int(m))"
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
