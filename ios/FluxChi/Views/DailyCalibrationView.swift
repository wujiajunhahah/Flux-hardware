import SwiftUI

struct DailyCalibrationView: View {
    var onFinished: (() -> Void)? = nil

    @EnvironmentObject var service: FluxService
    @EnvironmentObject var bleManager: BLEManager
    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) private var colorScheme

    @State private var phase: Phase = .prepare
    @State private var progress: CGFloat = 0
    @State private var relaxSamples: [[Double]] = []
    @State private var relaxMean: [Double] = Array(repeating: 0, count: 8)
    @State private var mvcMax: [Double] = Array(repeating: 0, count: 8)
    @State private var sampleTimer: Timer?
    @State private var summaryStore: EMGCalibrationStore?
    @State private var liveForceLevel: Double = 0

    private enum Phase: Int, CaseIterable, Equatable {
        case prepare = 0, relax = 1, mvc = 2, done = 3
    }

    private let relaxDuration: TimeInterval = 10
    private let mvcDuration: TimeInterval = 5
    private let sampleInterval: TimeInterval = 1.0 / 20.0

    // MARK: - Signal

    private var liveRMS: [Double] {
        if let r = service.state?.rms, !r.isEmpty { return pad8(r) }
        return pad8(bleManager.latestRMS)
    }

    private var activeCh: Int {
        max(1, bleManager.isConnected ? bleManager.activeChannelCount : 8)
    }

    private var hasSignal: Bool {
        service.isConnected || bleManager.isConnected
    }

    private var partialCalibration: EMGCalibrationStore? {
        guard phase == .mvc, !relaxSamples.isEmpty else { return nil }
        return EMGCalibrationStore(
            relaxMean: relaxMean,
            mvcPeak: mvcMax.map { max($0, 1) },
            quality: 0,
            calibratedAt: nil
        )
    }

    // MARK: - Theme

    private var immersiveBg: Color {
        colorScheme == .dark ? Flux.Surface.focusImmersive : Color(UIColor.systemBackground)
    }

    private var label1: Color {
        colorScheme == .dark ? .white : .primary
    }

    private var label2: Color {
        colorScheme == .dark ? .white.opacity(0.72) : .secondary
    }

    private var ringTheme: FluxRadialEMGRingView.Theme {
        colorScheme == .dark ? .dark : .light
    }

    private var tint: Color {
        colorScheme == .dark ? .cyan : Color.accentColor
    }

    // MARK: - Body

    var body: some View {
        NavigationStack {
            ZStack {
                immersiveBg.ignoresSafeArea()

                VStack(spacing: 0) {
                    stepIndicator
                        .padding(.top, 12)

                    Spacer(minLength: 16)

                    FluxRadialEMGRingView(
                        rms: liveRMS,
                        activeChannelCount: activeCh,
                        barCount: 144,
                        theme: ringTheme,
                        calibration: partialCalibration
                    )
                    .frame(maxHeight: 300)

                    if phase == .relax || phase == .mvc {
                        progressBar
                            .padding(.top, 16)
                            .transition(.opacity.combined(with: .offset(y: 8)))
                    }

                    Spacer(minLength: 16)

                    phaseContent
                        .padding(.bottom, 40)
                }
                .padding(.horizontal, Flux.Spacing.section)
            }
            .navigationTitle(phaseLabel)
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbarColorScheme(colorScheme == .dark ? .dark : .light, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    if phase != .done {
                        Button("取消") {
                            stopSampling()
                            dismiss()
                        }
                    }
                }
            }
        }
        .animation(.spring(response: 0.5, dampingFraction: 0.85), value: phase)
        .statusBarHidden(true)
        .onDisappear { stopSampling() }
    }

    // MARK: - Step Indicator

    private var stepIndicator: some View {
        HStack(spacing: 6) {
            ForEach(0..<4, id: \.self) { i in
                Capsule()
                    .fill(i <= phase.rawValue ? tint : label2.opacity(0.2))
                    .frame(width: i <= phase.rawValue ? 20 : 8, height: 4)
            }
        }
        .animation(.spring(response: 0.4, dampingFraction: 0.7), value: phase)
    }

    // MARK: - Progress Bar

    private var progressBar: some View {
        VStack(spacing: 8) {
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(activePhaseColor.opacity(0.15))
                    Capsule()
                        .fill(activePhaseColor)
                        .frame(width: geo.size.width * progress)
                        .animation(.linear(duration: sampleInterval), value: progress)
                }
            }
            .frame(height: 4)

            HStack {
                Text(remainingText)
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(label2)
                    .contentTransition(.numericText())
                Spacer()
                if phase == .mvc {
                    Text("力度 \(Int(liveForceLevel * 100))%")
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundStyle(activePhaseColor)
                        .contentTransition(.numericText())
                }
            }
        }
        .padding(.horizontal, 4)
    }

    private var activePhaseColor: Color {
        phase == .relax ? .green : .orange
    }

    private var remainingText: String {
        let dur = phase == .relax ? relaxDuration : mvcDuration
        let remaining = max(0, Int(ceil(dur * Double(1 - progress))))
        return "\(remaining)s"
    }

    // MARK: - Phase Content

    @ViewBuilder
    private var phaseContent: some View {
        switch phase {
        case .prepare:
            prepareView
                .transition(.asymmetric(
                    insertion: .opacity,
                    removal: .opacity.combined(with: .offset(y: -20))
                ))
        case .relax:
            activePhaseText(
                icon: "hand.raised",
                iconColor: .green,
                title: "放松手臂",
                subtitle: "自然下垂或轻放桌面，不要刻意用力"
            )
            .transition(.asymmetric(
                insertion: .opacity.combined(with: .offset(y: 20)),
                removal: .opacity.combined(with: .offset(y: -20))
            ))
        case .mvc:
            activePhaseText(
                icon: "hand.point.up.braille",
                iconColor: .orange,
                title: "最大力握拳",
                subtitle: "保持约 5 秒，越稳定越好"
            )
            .transition(.asymmetric(
                insertion: .opacity.combined(with: .offset(y: 20)),
                removal: .opacity.combined(with: .offset(y: -20))
            ))
        case .done:
            doneView
                .transition(.opacity.combined(with: .scale(scale: 0.95)))
        }
    }

    // MARK: - Prepare

    private var prepareView: some View {
        VStack(spacing: 18) {
            Text("约 15 秒，圆环会随手部用力变化。\n请先戴好手环并保持连接。")
                .font(.subheadline)
                .foregroundStyle(label2)
                .multilineTextAlignment(.center)
                .lineSpacing(3)

            if !hasSignal {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.caption)
                    Text("未检测到信号，请先连接设备")
                }
                .font(.caption)
                .foregroundStyle(.orange)
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(.orange.opacity(0.08), in: Capsule())
            }

            Button {
                beginRelax()
            } label: {
                Text("开始校准")
                    .font(.headline)
                    .foregroundStyle(colorScheme == .dark ? .black : .white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(tint, in: .rect(cornerRadius: 14))
            }
            .disabled(!hasSignal)
            .opacity(hasSignal ? 1 : 0.4)
            .padding(.top, 4)
        }
    }

    // MARK: - Active Phase Text

    private func activePhaseText(icon: String, iconColor: Color, title: String, subtitle: String) -> some View {
        VStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(iconColor.opacity(0.1))
                    .frame(width: 52, height: 52)
                Image(systemName: icon)
                    .font(.system(size: 22))
                    .foregroundStyle(iconColor)
                    .symbolEffect(.pulse, options: .repeating)
            }

            Text(title)
                .font(.title3.bold())
                .foregroundStyle(label1)

            Text(subtitle)
                .font(.subheadline)
                .foregroundStyle(label2)
                .multilineTextAlignment(.center)
        }
    }

    // MARK: - Done

    private var doneView: some View {
        VStack(spacing: 18) {
            ZStack {
                Circle()
                    .fill(tint.opacity(0.12))
                    .frame(width: 64, height: 64)
                Image(systemName: "checkmark")
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundStyle(tint)
            }

            Text("校准完成")
                .font(.title2.bold())
                .foregroundStyle(label1)

            if let store = summaryStore {
                qualityBadge(store.quality)

                Text(qualityCopy(store.quality))
                    .font(.subheadline)
                    .foregroundStyle(label2)
                    .multilineTextAlignment(.center)
                    .lineSpacing(3)
            }

            Button {
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                onFinished?()
                dismiss()
            } label: {
                Text("完成")
                    .font(.headline)
                    .foregroundStyle(colorScheme == .dark ? .black : .white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(tint, in: .rect(cornerRadius: 14))
            }
            .padding(.top, 4)
        }
    }

    private func qualityBadge(_ q: Int) -> some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { i in
                Image(systemName: i < q ? "star.fill" : "star")
                    .font(.system(size: 14))
                    .foregroundStyle(i < q ? tint : label2.opacity(0.3))
            }
        }
    }

    // MARK: - Phase Logic

    private func beginRelax() {
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        relaxSamples.removeAll()
        progress = 0
        phase = .relax
        startSampling(duration: relaxDuration, onTick: {
            relaxSamples.append(pad8(liveRMS))
        }, onComplete: {
            computeRelaxMean()
            beginMVC()
        })
    }

    private func beginMVC() {
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        mvcMax = Array(repeating: 0, count: 8)
        progress = 0
        liveForceLevel = 0
        phase = .mvc
        startSampling(duration: mvcDuration, onTick: {
            let v = liveRMS
            for i in 0..<8 {
                mvcMax[i] = max(mvcMax[i], i < v.count ? v[i] : 0)
            }
            updateForceLevel(v)
        }, onComplete: {
            finishCalibration()
        })
    }

    private func computeRelaxMean() {
        let n = EMGCalibrationStore.channelCount
        var mean = Array(repeating: 0.0, count: n)
        guard !relaxSamples.isEmpty else {
            relaxMean = mean
            return
        }
        for ch in 0..<n {
            let vals = relaxSamples.map { ch < $0.count ? $0[ch] : 0 }
            mean[ch] = vals.reduce(0, +) / Double(vals.count)
        }
        relaxMean = mean
    }

    private func updateForceLevel(_ current: [Double]) {
        let n = min(current.count, relaxMean.count, 8)
        guard n > 0 else { return }
        var total: Double = 0
        for i in 0..<n {
            let span = max(mvcMax[i] - relaxMean[i], 1)
            let level = max(0, (current[i] - relaxMean[i]) / span)
            total += min(1, level)
        }
        let avg = total / Double(n)
        liveForceLevel = liveForceLevel * 0.7 + avg * 0.3
    }

    private func finishCalibration() {
        stopSampling()
        let n = EMGCalibrationStore.channelCount
        var mvc = mvcMax
        for i in 0..<n {
            mvc[i] = max(mvc[i], relaxMean[i] + 1)
        }
        let q = EMGCalibrationStore.computeQuality(relax: relaxMean, mvc: mvc)
        let store = EMGCalibrationStore(relaxMean: relaxMean, mvcPeak: mvc, quality: q, calibratedAt: nil)
        store.save()
        summaryStore = store
        progress = 1
        phase = .done

        UINotificationFeedbackGenerator()
            .notificationOccurred(q >= 2 ? .success : .warning)
    }

    // MARK: - Timer

    private func startSampling(duration: TimeInterval, onTick: @escaping () -> Void, onComplete: @escaping () -> Void) {
        stopSampling()
        let start = Date()
        sampleTimer = Timer.scheduledTimer(withTimeInterval: sampleInterval, repeats: true) { t in
            Task { @MainActor in
                let elapsed = Date().timeIntervalSince(start)
                onTick()
                progress = min(1, CGFloat(elapsed / duration))
                if elapsed >= duration {
                    t.invalidate()
                    sampleTimer = nil
                    progress = 1
                    onComplete()
                }
            }
        }
        if let timer = sampleTimer {
            RunLoop.main.add(timer, forMode: .common)
        }
    }

    private func stopSampling() {
        sampleTimer?.invalidate()
        sampleTimer = nil
    }

    // MARK: - Helpers

    private var phaseLabel: String {
        switch phase {
        case .prepare: return "每日校准"
        case .relax: return "放松采样"
        case .mvc: return "用力采样"
        case .done: return "校准完成"
        }
    }

    private func pad8(_ v: [Double]) -> [Double] {
        var a = Array(v.prefix(8))
        while a.count < 8 { a.append(0) }
        return a
    }

    private func qualityCopy(_ q: Int) -> String {
        switch q {
        case 3: return "信号对比清晰，已记住你安静与用力时的差异，\n续航估算会更贴你的习惯。"
        case 2: return "记录成功。若下次戴环位置不同，可再校准一次。"
        case 1: return "已保存。试试握拳再用力一点，\n或检查手环是否戴紧。"
        default: return "已保存基础数据。\n建议重新校准并确保握拳时用力更明显。"
        }
    }
}

#Preview {
    DailyCalibrationView()
        .environmentObject(FluxService())
        .environmentObject(BLEManager())
}
