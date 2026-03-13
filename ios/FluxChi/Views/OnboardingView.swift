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

            nextButton("允许权限") {
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

    @State private var isCalibrating = false
    @State private var calibrationProgress: CGFloat = 0
    @State private var calibrationTimer: Timer?
    @State private var calibrationDone = false

    private var calibrationStep: some View {
        VStack(spacing: 32) {
            Spacer()

            ZStack {
                Circle()
                    .stroke(Color.green.opacity(0.2), lineWidth: 6)
                    .frame(width: 140, height: 140)

                Circle()
                    .trim(from: 0, to: calibrationProgress)
                    .stroke(Color.green, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .frame(width: 140, height: 140)
                    .animation(.linear(duration: 0.1), value: calibrationProgress)

                if calibrationDone {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 56))
                        .foregroundStyle(.green)
                } else {
                    Image(systemName: "hand.point.down.fill")
                        .font(.system(size: 44))
                        .foregroundStyle(isCalibrating ? .green : .secondary)
                        .symbolEffect(.pulse, isActive: isCalibrating)
                }
            }

            VStack(spacing: 12) {
                Text(calibrationDone ? "校准完成" : "基线校准")
                    .font(.title.bold())

                Text(calibrationDone
                     ? "已记录你的静息基线，开始专注吧！"
                     : "放松手臂，保持自然姿势 10 秒\nFocuX 将记录你的肌肉静息基线")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            Spacer()

            stepIndicator(current: 3, total: 4)

            if calibrationDone {
                nextButton("开始使用") {
                    isCompleted = true
                }
            } else if isCalibrating {
                Text("请保持静止…")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .padding(.bottom, 50)
            } else {
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

    // MARK: - Calibration Logic

    private func startCalibration() {
        isCalibrating = true
        calibrationProgress = 0

        let totalDuration: TimeInterval = 10
        let interval: TimeInterval = 0.1
        let increment = CGFloat(interval / totalDuration)

        calibrationTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { timer in
            Task { @MainActor in
                calibrationProgress += increment
                if calibrationProgress >= 1.0 {
                    timer.invalidate()
                    calibrationTimer = nil
                    isCalibrating = false
                    calibrationDone = true

                    // 存储今日已校准
                    UserDefaults.standard.set(
                        Date().timeIntervalSince1970,
                        forKey: "flux_last_calibration"
                    )
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
