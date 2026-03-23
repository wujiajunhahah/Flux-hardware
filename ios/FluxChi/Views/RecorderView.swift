import SwiftUI
import SwiftData

struct RecorderView: View {
    @EnvironmentObject var service: FluxService
    @EnvironmentObject var bleManager: BLEManager
    @EnvironmentObject var sessionManager: SessionManager
    @Environment(\.modelContext) private var modelContext

    @State private var showSegmentPicker = false
    @State private var showFeedback = false
    @State private var finishedSession: Session?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: Flux.Spacing.section) {
                    if sessionManager.isRecording {
                        recordingHeader
                        liveStaminaView
                        segmentInfo
                        liveEMGView
                    } else {
                        idleView
                    }
                }
                .padding()
            }
            .navigationTitle("记录")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    FluxLiveIndicator(isLive: service.isConnected || bleManager.isConnected)
                }
            }
            .sheet(isPresented: $showFeedback) {
                if let session = finishedSession {
                    FeedbackView(session: session)
                }
            }
        }
    }

    // MARK: - Idle

    private var idleView: some View {
        VStack(spacing: Flux.Spacing.section) {
            Spacer().frame(height: 40)

            Image(systemName: "waveform.badge.plus")
                .font(.system(size: 60))
                .foregroundStyle(.secondary)
                .symbolEffect(.pulse)

            Text("准备就绪")
                .font(.title2)
                .fontWeight(.semibold)

            Text("连接手环后点击开始记录")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            if !service.isConnected && !bleManager.isConnected {
                FluxStatusBadge(label: "未连接", icon: "wifi.slash", tint: .orange)
            } else {
                FluxStatusBadge(
                    label: bleManager.isConnected ? "BLE 已连接" : "WiFi 已连接",
                    icon: bleManager.isConnected ? "antenna.radiowaves.left.and.right" : "wifi",
                    tint: Flux.Colors.success
                )
            }

            Spacer().frame(height: 20)
            startButton
        }
    }

    // MARK: - Recording Header

    private var recordingHeader: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(Flux.Colors.accent)
                        .frame(width: 10, height: 10)
                        .opacity(sessionManager.isPaused ? 0.3 : 1)

                    Text(sessionManager.isPaused ? "已暂停" : "录制中")
                        .font(.headline)
                }

                Text(Flux.formatDuration(sessionManager.elapsed))
                    .font(Flux.Typography.metric(28))
                    .contentTransition(.numericText())
                    .animation(.linear, value: Int(sessionManager.elapsed))
            }

            Spacer()
            recordingControls
        }
        .padding()
        .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.large))
    }

    // MARK: - Live Stamina

    private var liveStaminaView: some View {
        let val = service.personalizedDisplayStamina
        let st = service.displayStaminaState
        return StaminaRingView(value: val, state: st)
    }

    // MARK: - Segment Info

    @ViewBuilder
    private var segmentInfo: some View {
        if let seg = sessionManager.activeSegment {
            HStack {
                Label(seg.label.displayName, systemImage: seg.label.icon)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundStyle(seg.label.color)

                Spacer()

                Text(Flux.formatDuration(seg.duration))
                    .font(Flux.Typography.mono)
                    .foregroundStyle(.secondary)

                Button {
                    showSegmentPicker = true
                } label: {
                    Image(systemName: "rectangle.split.1x2")
                        .font(.title3)
                }
                .confirmationDialog("新建分段", isPresented: $showSegmentPicker) {
                    ForEach(SegmentLabel.allCases) { label in
                        Button(label.displayName) {
                            sessionManager.addSegment(label: label)
                        }
                    }
                }
            }
            .padding()
            .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))
        }
    }

    // MARK: - Live EMG

    @ViewBuilder
    private var liveEMGView: some View {
        if service.isConnected || bleManager.isConnected {
            let rms = liveRMSVector
            VStack(alignment: .leading, spacing: Flux.Spacing.item) {
                FluxSectionLabel(title: "手势与力度", icon: "waveform")
                FluxRadialEMGRingView(
                    rms: rms,
                    activeChannelCount: bleManager.isConnected ? bleManager.activeChannelCount : 8,
                    barCount: 128,
                    theme: .light,
                    calibration: EMGCalibrationStore.load()
                )
                .frame(height: 220)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity)
                .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))
                if let raw = service.state?.rms, !raw.isEmpty {
                    FluxEMGBars(rms: raw)
                        .padding()
                        .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))
                }
            }
        }
    }

    private var liveRMSVector: [Double] {
        if let r = service.state?.rms, !r.isEmpty {
            return padRMS(r)
        }
        return padRMS(bleManager.latestRMS)
    }

    private func padRMS(_ v: [Double]) -> [Double] {
        var a = Array(v.prefix(8))
        while a.count < 8 { a.append(0) }
        return a
    }

    // MARK: - Controls

    private var startButton: some View {
        Button {
            let source: SessionSource = bleManager.isConnected ? .ble : .wifi
            sessionManager.startSession(source: source)
        } label: {
            HStack(spacing: 12) {
                Image(systemName: "record.circle")
                    .font(.title2)
                Text("开始记录")
                    .font(.headline)
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(Flux.Colors.accent, in: .rect(cornerRadius: Flux.Radius.large))
        }
        .disabled(!service.isConnected && !bleManager.isConnected)
    }

    private var recordingControls: some View {
        HStack(spacing: 16) {
            Button {
                if sessionManager.isPaused {
                    sessionManager.resumeSession()
                } else {
                    sessionManager.pauseSession()
                }
            } label: {
                Image(systemName: sessionManager.isPaused ? "play.fill" : "pause.fill")
                    .font(.title2)
                    .foregroundStyle(.primary)
                    .frame(width: 44, height: 44)
                    .background(.ultraThinMaterial, in: Circle())
            }

            Button {
                if let session = sessionManager.endSession() {
                    let summary = SummaryEngine.generate(for: session)
                    SummaryEngine.apply(summary, to: session)
                    modelContext.saveLogged()
                    finishedSession = session
                    showFeedback = true
                }
            } label: {
                Image(systemName: "stop.fill")
                    .font(.title2)
                    .foregroundStyle(.white)
                    .frame(width: 44, height: 44)
                    .background(Flux.Colors.accent, in: Circle())
            }
        }
    }
}
