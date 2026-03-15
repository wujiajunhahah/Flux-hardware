import SwiftUI
import CoreBluetooth

struct OnboardingView: View {
    @EnvironmentObject var bleManager: BLEManager
    @EnvironmentObject var alertManager: AlertManager
    @Binding var isCompleted: Bool

    @State private var currentStep = 0

    var body: some View {
        ZStack {
            Color(.systemBackground).ignoresSafeArea()

            TabView(selection: $currentStep) {
                welcomeStep.tag(0)
                permissionStep.tag(1)
                deviceStep.tag(2)
                calibrationStep.tag(3)
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
            .animation(.spring(duration: 0.4), value: currentStep)
        }
    }

    // MARK: - OB1: 欢迎

    private var welcomeStep: some View {
        VStack(spacing: 32) {
            Spacer()

            Image(systemName: "hand.raised.fingers.spread.fill")
                .font(.system(size: 72))
                .foregroundStyle(Flux.Colors.accent.gradient)

            VStack(spacing: 12) {
                Text("欢迎使用 FocuX")
                    .font(.largeTitle.bold())

                Text("基于 EMG 生物信号的智能专注伴侣")
                    .font(.title3)
                    .foregroundStyle(.secondary)

                Text("通过手环实时感知你的身体状态，\n帮你科学管理专注与休息。")
                    .font(.subheadline)
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
                    .padding(.top, 4)
            }

            Spacer()

            stepIndicator(current: 0, total: 4)

            nextButton("开始设置") {
                currentStep = 1
            }
        }
        .padding(Flux.Spacing.section)
    }

    // MARK: - OB2: 权限

    private var permissionStep: some View {
        VStack(spacing: 32) {
            Spacer()

            Image(systemName: "bell.badge.fill")
                .font(.system(size: 64))
                .foregroundStyle(Color(.systemOrange).gradient)

            VStack(spacing: 12) {
                Text("开启权限")
                    .font(.title.bold())

                Text("FocuX 需要以下权限来正常工作")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            VStack(spacing: 16) {
                permissionRow(
                    icon: "antenna.radiowaves.left.and.right",
                    title: "蓝牙",
                    desc: "连接 WAVELETECH 手环",
                    tint: Color(.systemTeal)
                )
                permissionRow(
                    icon: "bell.fill",
                    title: "通知",
                    desc: "休息提醒和专注状态更新",
                    tint: Color(.systemOrange)
                )
            }
            .padding()
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: Flux.Radius.large))

            Spacer()

            stepIndicator(current: 1, total: 4)

            nextButton("继续") {
                alertManager.requestPermission()
                currentStep = 2
            }
        }
        .padding(Flux.Spacing.section)
    }

    // MARK: - OB3: 设备配对

    private var deviceStep: some View {
        VStack(spacing: 32) {
            Spacer()

            Image(systemName: "waveform.badge.magnifyingglass")
                .font(.system(size: 64))
                .foregroundStyle(Color(.systemTeal).gradient)

            VStack(spacing: 12) {
                Text("连接手环")
                    .font(.title.bold())

                Text("请确保 WAVELETECH 手环已开启")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            VStack(spacing: 12) {
                if bleManager.isScanning {
                    HStack(spacing: 12) {
                        ProgressView()
                        Text("正在搜索设备…")
                            .foregroundStyle(.secondary)
                    }
                    .padding()
                }

                ForEach(bleManager.discoveredDevices, id: \.identifier) { device in
                    Button {
                        bleManager.connect(to: device)
                    } label: {
                        HStack {
                            Image(systemName: "waveform.badge.plus")
                                .foregroundStyle(Color(.systemTeal))
                            Text(device.name ?? "未知设备")
                                .foregroundStyle(.primary)
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                        .padding()
                        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: Flux.Radius.medium))
                    }
                    .buttonStyle(.plain)
                }

                if bleManager.isConnected {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text("已连接 \(bleManager.connectedDeviceName ?? "")")
                            .font(.subheadline.weight(.semibold))
                    }
                    .padding()
                }

                if !bleManager.isScanning && !bleManager.isConnected {
                    Button {
                        bleManager.startScan()
                    } label: {
                        Label("重新扫描", systemImage: "arrow.clockwise")
                            .font(.subheadline)
                    }
                }
            }
            .padding()
            .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: Flux.Radius.large))

            Spacer()

            stepIndicator(current: 2, total: 4)

            VStack(spacing: 12) {
                nextButton(bleManager.isConnected ? "下一步" : "稍后连接") {
                    currentStep = 3
                }
            }
        }
        .padding(Flux.Spacing.section)
        .onAppear {
            if !bleManager.isConnected {
                bleManager.startScan()
            }
        }
    }

    // MARK: - OB4: 基线校准

    enum CalibrationPhase { case idle, baseline, mvc, done }

    @State private var calibrationPhase: CalibrationPhase = .idle
    @State private var calibrationProgress: CGFloat = 0
    @State private var calibrationTimer: Timer?

    private var calibrationStep: some View {
        let ringColor: Color = calibrationPhase == .mvc ? .orange : .green

        return VStack(spacing: 32) {
            Spacer()

            ZStack {
                Circle()
                    .stroke(ringColor.opacity(0.2), lineWidth: 6)
                    .frame(width: 140, height: 140)

                Circle()
                    .trim(from: 0, to: calibrationProgress)
                    .stroke(ringColor, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .frame(width: 140, height: 140)
                    .animation(.linear(duration: 0.1), value: calibrationProgress)

                switch calibrationPhase {
                case .done:
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 56))
                        .foregroundStyle(.green)
                case .mvc:
                    Image(systemName: "hand.raised.fill")
                        .font(.system(size: 44))
                        .foregroundStyle(.orange)
                        .symbolEffect(.pulse, isActive: true)
                case .baseline:
                    Image(systemName: "hand.point.down.fill")
                        .font(.system(size: 44))
                        .foregroundStyle(.green)
                        .symbolEffect(.pulse, isActive: true)
                case .idle:
                    Image(systemName: "hand.point.down.fill")
                        .font(.system(size: 44))
                        .foregroundStyle(.secondary)
                }
            }

            VStack(spacing: 12) {
                Text(calibrationTitle)
                    .font(.title.bold())

                Text(calibrationSubtitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            Spacer()

            stepIndicator(current: 3, total: 4)

            switch calibrationPhase {
            case .done:
                nextButton("开始使用") {
                    isCompleted = true
                }
            case .baseline, .mvc:
                Text(calibrationPhase == .baseline ? "请保持放松…" : "最大力握拳！")
                    .font(.subheadline)
                    .foregroundStyle(calibrationPhase == .mvc ? .orange : .secondary)
                    .padding(.bottom, 50)
            case .idle:
                VStack(spacing: 12) {
                    nextButton("开始校准") {
                        startCalibration()
                    }
                    .disabled(!bleManager.isConnected)

                    if !bleManager.isConnected {
                        Button("跳过校准") {
                            isCompleted = true
                        }
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .padding(Flux.Spacing.section)
    }

    private var calibrationTitle: String {
        switch calibrationPhase {
        case .idle:     return "基线校准"
        case .baseline: return "静息基线"
        case .mvc:      return "最大发力"
        case .done:     return "校准完成"
        }
    }

    private var calibrationSubtitle: String {
        switch calibrationPhase {
        case .idle:
            return "两步校准：放松手臂 10 秒 + 最大握拳 5 秒\nFocuX 将记录你的肌肉基线范围"
        case .baseline:
            return "放松手臂，保持自然姿势\n正在记录静息基线…"
        case .mvc:
            return "请尽全力握拳并保持 5 秒\n正在记录最大发力值…"
        case .done:
            return "已记录静息基线和最大发力值\n开始你的专注旅程吧！"
        }
    }

    // MARK: - Calibration Logic

    private func startCalibration() {
        // Phase 1: 静息基线 10 秒
        calibrationPhase = .baseline
        calibrationProgress = 0

        let baselineDuration: TimeInterval = 10
        let interval: TimeInterval = 0.1
        let baselineInc = CGFloat(interval / baselineDuration)

        calibrationTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { timer in
            Task { @MainActor in
                calibrationProgress += baselineInc
                if calibrationProgress >= 1.0 {
                    timer.invalidate()
                    calibrationTimer = nil
                    startMVCPhase()
                }
            }
        }
    }

    private func startMVCPhase() {
        // Phase 2: 最大发力 5 秒
        calibrationPhase = .mvc
        calibrationProgress = 0

        let mvcDuration: TimeInterval = 5
        let interval: TimeInterval = 0.1
        let mvcInc = CGFloat(interval / mvcDuration)

        calibrationTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { timer in
            Task { @MainActor in
                calibrationProgress += mvcInc
                if calibrationProgress >= 1.0 {
                    timer.invalidate()
                    calibrationTimer = nil
                    calibrationPhase = .done

                    // 存储今日已校准（含 MVC）
                    UserDefaults.standard.set(
                        Date().timeIntervalSince1970,
                        forKey: "flux_last_calibration"
                    )
                    UserDefaults.standard.set(true, forKey: "flux_mvc_collected")
                }
            }
        }
    }

    // MARK: - Shared Components

    private func nextButton(_ title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(Flux.Colors.accent.gradient, in: .rect(cornerRadius: Flux.Radius.large))
                .foregroundStyle(.white)
        }
    }

    private func stepIndicator(current: Int, total: Int) -> some View {
        HStack(spacing: 8) {
            ForEach(0..<total, id: \.self) { i in
                Capsule()
                    .fill(i <= current ? Flux.Colors.accent : Color.primary.opacity(0.15))
                    .frame(width: i == current ? 24 : 8, height: 8)
                    .animation(.spring(duration: 0.3), value: current)
            }
        }
    }

    private func permissionRow(icon: String, title: String, desc: String, tint: Color) -> some View {
        HStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(tint.opacity(0.12))
                    .frame(width: 40, height: 40)
                Image(systemName: icon)
                    .foregroundStyle(tint)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Text(desc)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
    }
}

#Preview {
    OnboardingView(isCompleted: .constant(false))
        .environmentObject(BLEManager())
        .environmentObject(AlertManager())
}
