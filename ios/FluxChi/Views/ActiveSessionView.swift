import SwiftUI
import SwiftData

struct ActiveSessionView: View {
    @EnvironmentObject var service: FluxService
    @EnvironmentObject var sessionManager: SessionManager
    @EnvironmentObject var bleManager: BLEManager
    @EnvironmentObject var liveActivityManager: LiveActivityManager
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var modelContext

    /// 结束 session 后的回调，传出已完成的 Session
    var onSessionFinished: ((Session) -> Void)?

    // MARK: - State

    @State private var holdProgress: CGFloat = 0
    @State private var isHolding = false
    @State private var breatheScale: CGFloat = 1.0
    @State private var isPausedLocal = false
    @State private var holdTimer: Timer?

    // 休息提醒
    @State private var showRestBanner = false
    @State private var restBannerDismissed = false

    // BLE 断连自动暂停
    @State private var bleDisconnectedAt: Date?
    @State private var bleReconnectTimer: Timer?
    @State private var showDisconnectBanner = false

    // 佩戴脱落
    @State private var showDetachBanner = false

    // 休息模式
    @State private var isResting = false
    @State private var restDuration: TimeInterval = 5 * 60  // 默认 5 分钟
    @State private var restRemaining: TimeInterval = 5 * 60
    @State private var restTimer: Timer?
    @State private var showRestEndDialog = false

    private var stamina: StaminaData? { service.state?.stamina }
    private var staminaValue: Double { service.personalizedStaminaValue }
    private var staminaState: StaminaState {
        StaminaState(rawValue: stamina?.state ?? "focused") ?? .focused
    }

    // MARK: - Body

    var body: some View {
        ZStack {
            // 背景：专注=黑色，休息=深绿
            (isResting ? Color(red: 0.04, green: 0.12, blue: 0.08) : Color.black)
                .ignoresSafeArea()
                .animation(.easeInOut(duration: 0.8), value: isResting)

            VStack(spacing: 0) {
                topStatusBar
                    .padding(.top, 16)

                // BLE 断连 Banner
                if showDisconnectBanner {
                    disconnectBanner
                        .transition(.move(edge: .top).combined(with: .opacity))
                        .padding(.top, 8)
                }

                // 佩戴脱落 Banner
                if showDetachBanner && !showDisconnectBanner {
                    detachBanner
                        .transition(.move(edge: .top).combined(with: .opacity))
                        .padding(.top, 8)
                }

                // 休息提醒 Banner
                if showRestBanner && !restBannerDismissed && !isResting && !showDisconnectBanner && !showDetachBanner {
                    restAlertBanner
                        .transition(.move(edge: .top).combined(with: .opacity))
                        .padding(.top, 8)
                }

                Spacer()

                if isResting {
                    restCountdownSection
                } else {
                    staminaRingSection
                }

                Spacer()

                if isResting {
                    restStatusLabel
                        .padding(.bottom, 32)
                } else {
                    statusLabel
                        .padding(.bottom, 32)
                }

                controlBar
                    .padding(.bottom, 40)
            }
            .padding(.horizontal, Flux.Spacing.section)
        }
        .preferredColorScheme(.dark)
        .statusBarHidden()
        .onAppear {
            startBreathingAnimation()
            UIApplication.shared.isIdleTimerDisabled = true
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
        }
        .onChange(of: sessionManager.isPaused) { _, paused in
            isPausedLocal = paused
            if paused {
                withAnimation(.easeOut(duration: 0.6)) { breatheScale = 1.0 }
            } else {
                startBreathingAnimation()
            }
        }
        .onChange(of: staminaValue) { _, newVal in
            evaluateRestReminder(stamina: newVal)
        }
        .alert("休息结束", isPresented: $showRestEndDialog) {
            Button("继续专注") { exitRestMode(endSession: false) }
            Button("结束本次工作") { exitRestMode(endSession: true) }
        } message: {
            Text("休息时间到了，你想继续专注还是结束本次工作？")
        }
        // BLE 断连自动暂停
        .onChange(of: bleManager.isConnected) { _, connected in
            handleBLEConnectionChange(connected)
        }
        // 佩戴脱落检测
        .onReceive(NotificationCenter.default.publisher(for: .bleDeviceDetached)) { _ in
            handleDeviceDetach()
        }
        // 通知点击触发休息 Banner
        .onReceive(NotificationCenter.default.publisher(for: FluxChiApp.showRestFromNotification)) { _ in
            if !isResting {
                withAnimation(.spring(duration: 0.5)) {
                    showRestBanner = true
                    restBannerDismissed = false
                }
            }
        }
    }

    // MARK: - Top Status Bar

    private var topStatusBar: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(isResting ? .green : (sessionManager.isPaused ? .orange : .green))
                .frame(width: 8, height: 8)

            Text(isResting ? "休息中" : (sessionManager.isPaused ? "已暂停" : "专注中"))
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.secondary)

            Text("·")
                .foregroundStyle(.secondary)

            Text(formatElapsed(sessionManager.elapsed))
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .foregroundStyle(.secondary)
                .contentTransition(.numericText())
                .animation(.easeInOut, value: Int(sessionManager.elapsed))

            Spacer()
        }
    }

    // MARK: - Rest Alert Banner

    private var restAlertBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: "leaf.fill")
                .foregroundStyle(.green)

            VStack(alignment: .leading, spacing: 2) {
                Text("身体需要休息")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)
                Text("续航值较低，建议休息 5 分钟")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
            }

            Spacer()

            Button("进入休息") {
                withAnimation(.spring(duration: 0.5)) {
                    enterRestMode()
                }
            }
            .font(.caption.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.green, in: Capsule())

            Button {
                withAnimation { restBannerDismissed = true }
            } label: {
                Image(systemName: "xmark")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
        .padding(14)
        .background(Color.white.opacity(0.1), in: .rect(cornerRadius: 14))
        .sensoryFeedback(.warning, trigger: showRestBanner)
    }

    // MARK: - Stamina Ring (专注模式)

    private var staminaRingSection: some View {
        ZStack {
            Circle()
                .stroke(
                    staminaRingColor.opacity(isPausedLocal ? 0.05 : 0.15),
                    lineWidth: 1.5
                )
                .frame(width: 300, height: 300)
                .scaleEffect(breatheScale)

            StaminaRingView(
                value: staminaValue,
                state: staminaState,
                size: 280
            )
            .drawingGroup()
            .saturation(isPausedLocal ? 0.3 : 1.0)
            .animation(.easeInOut(duration: 0.4), value: isPausedLocal)
        }
    }

    // MARK: - Rest Countdown (休息模式)

    private var restCountdownSection: some View {
        ZStack {
            // 外环（倒计时进度）
            Circle()
                .stroke(Color.green.opacity(0.15), lineWidth: 8)
                .frame(width: 260, height: 260)

            Circle()
                .trim(from: 0, to: restProgress)
                .stroke(
                    Color.green,
                    style: StrokeStyle(lineWidth: 8, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
                .frame(width: 260, height: 260)
                .animation(.linear(duration: 1), value: restProgress)

            VStack(spacing: 8) {
                Image(systemName: "leaf.fill")
                    .font(.system(size: 32))
                    .foregroundStyle(.green)

                Text(formatElapsed(restRemaining))
                    .font(.system(size: 48, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                    .contentTransition(.numericText())
                    .animation(.easeInOut, value: Int(restRemaining))

                Text("放松一下")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.green.opacity(0.8))
            }
        }
    }

    private var restProgress: CGFloat {
        guard restDuration > 0 else { return 0 }
        return CGFloat(1.0 - restRemaining / restDuration)
    }

    // MARK: - Status Labels

    private var statusLabel: some View {
        VStack(spacing: 6) {
            Text(staminaState.displayName)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(staminaRingColor)

            Text("连续 \(Int(sessionManager.elapsed / 60)) min")
                .font(.system(size: 13, design: .monospaced))
                .foregroundStyle(.secondary)
                .contentTransition(.numericText())
                .animation(.easeInOut, value: Int(sessionManager.elapsed / 60))
        }
    }

    private var restStatusLabel: some View {
        VStack(spacing: 6) {
            Text("恢复中")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(.green)

            Text("续航 \(Int(staminaValue))")
                .font(.system(size: 13, design: .monospaced))
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Control Bar

    private var controlBar: some View {
        HStack(spacing: 48) {
            if isResting {
                // 休息模式：跳过休息按钮
                Button {
                    restTimer?.invalidate()
                    restTimer = nil
                    showRestEndDialog = true
                } label: {
                    Text("跳过")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.6))
                        .padding(.horizontal, 20)
                        .padding(.vertical, 12)
                        .background(Color.white.opacity(0.1), in: Capsule())
                }
            } else {
                // 专注模式：暂停/恢复
                Button {
                    if sessionManager.isPaused {
                        sessionManager.resumeSession()
                    } else {
                        sessionManager.pauseSession()
                    }
                } label: {
                    Image(systemName: sessionManager.isPaused ? "play.circle.fill" : "pause.circle.fill")
                        .font(.system(size: 56))
                        .foregroundStyle(.white.opacity(0.85))
                        .contentTransition(.symbolEffect(.replace))
                }

                // 长按结束
                endSessionButton
            }
        }
    }

    // MARK: - End Session (Long Press)

    private var endSessionButton: some View {
        ZStack {
            Circle()
                .stroke(Color.white.opacity(0.15), lineWidth: 4)
                .frame(width: 64, height: 64)

            Circle()
                .trim(from: 0, to: holdProgress)
                .stroke(
                    Flux.Colors.accent,
                    style: StrokeStyle(lineWidth: 4, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
                .frame(width: 64, height: 64)
                .animation(.linear(duration: 0.05), value: holdProgress)

            Image(systemName: "stop.fill")
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(holdProgress > 0 ? Flux.Colors.accent : .white.opacity(0.6))
        }
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    if !isHolding { beginHold() }
                }
                .onEnded { _ in
                    cancelHold()
                }
        )
        .sensoryFeedback(.impact(weight: .medium), trigger: isHolding)
        .accessibilityLabel("长按结束专注")
        .accessibilityHint("按住 1.5 秒结束本次专注")
    }

    // MARK: - Hold Logic

    private func beginHold() {
        isHolding = true
        holdProgress = 0

        let interval: TimeInterval = 0.02
        let totalDuration: TimeInterval = 1.5
        let increment = CGFloat(interval / totalDuration)

        holdTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { timer in
            Task { @MainActor in
                holdProgress += increment
                if holdProgress >= 1.0 {
                    timer.invalidate()
                    holdTimer = nil
                    finishSession()
                }
            }
        }
    }

    private func cancelHold() {
        isHolding = false
        holdTimer?.invalidate()
        holdTimer = nil
        withAnimation(.easeOut(duration: 0.3)) { holdProgress = 0 }
    }

    private func finishSession() {
        isHolding = false
        holdProgress = 1.0

        // 清理休息 timer
        restTimer?.invalidate()
        restTimer = nil

        liveActivityManager.endActivity()

        if let session = sessionManager.endSession() {
            let summary = SummaryEngine.generate(for: session)
            SummaryEngine.apply(summary, to: session)
            try? modelContext.save()
            onSessionFinished?(session)
        }

        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(.success)

        dismiss()
    }

    // MARK: - Rest Reminder Logic

    private func evaluateRestReminder(stamina: Double) {
        let workMinutes = sessionManager.elapsed / 60
        // 触发条件：stamina < 30 且连续工作 > 45 分钟
        if stamina < 30 && workMinutes > 45 && !isResting && !restBannerDismissed {
            withAnimation(.spring(duration: 0.5)) {
                showRestBanner = true
            }
        }
    }

    // MARK: - Disconnect Banner

    private var disconnectBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: "antenna.radiowaves.left.and.right.slash")
                .foregroundStyle(.red)

            VStack(alignment: .leading, spacing: 2) {
                Text("手环已断开")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)
                Text("已自动暂停，5 分钟内未重连将结束本次专注")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
            }

            Spacer()
        }
        .padding(14)
        .background(Color.red.opacity(0.2), in: .rect(cornerRadius: 14))
    }

    // MARK: - Detach Banner

    private var detachBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: "hand.raised.slash.fill")
                .foregroundStyle(.orange)

            VStack(alignment: .leading, spacing: 2) {
                Text("手环可能脱落")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)
                Text("信号持续极低，请检查佩戴位置")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
            }

            Spacer()

            Button("知道了") {
                withAnimation { showDetachBanner = false }
            }
            .font(.caption.weight(.semibold))
            .foregroundStyle(.white.opacity(0.6))
        }
        .padding(14)
        .background(Color.orange.opacity(0.2), in: .rect(cornerRadius: 14))
    }

    // MARK: - BLE Disconnect Logic

    private func handleBLEConnectionChange(_ connected: Bool) {
        if !connected {
            // 断连 → 自动暂停
            if !sessionManager.isPaused {
                sessionManager.pauseSession()
            }
            bleDisconnectedAt = Date()
            withAnimation { showDisconnectBanner = true }

            // 5 分钟超时自动结束
            bleReconnectTimer = Timer.scheduledTimer(withTimeInterval: 300, repeats: false) { _ in
                Task { @MainActor in
                    finishSession()
                }
            }
        } else {
            // 重连 → 自动恢复
            bleReconnectTimer?.invalidate()
            bleReconnectTimer = nil
            bleDisconnectedAt = nil
            withAnimation { showDisconnectBanner = false }

            if sessionManager.isPaused {
                sessionManager.resumeSession()
            }
        }
    }

    // MARK: - Device Detach Logic

    private func handleDeviceDetach() {
        guard !sessionManager.isPaused else { return }
        sessionManager.pauseSession()
        withAnimation { showDetachBanner = true }
    }

    // MARK: - Rest Mode

    private func enterRestMode() {
        isResting = true
        showRestBanner = false
        restBannerDismissed = true

        // 自动打 rest 分段
        sessionManager.addSegment(label: .rest)

        // 根据建议设定休息时长
        let suggested = stamina?.suggestedBreakMin ?? 5
        restDuration = max(suggested, 3) * 60
        restRemaining = restDuration

        // 启动倒计时
        restTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { timer in
            Task { @MainActor in
                restRemaining -= 1
                if restRemaining <= 0 {
                    timer.invalidate()
                    restTimer = nil
                    showRestEndDialog = true
                }
            }
        }
    }

    private func exitRestMode(endSession: Bool) {
        isResting = false
        restTimer?.invalidate()
        restTimer = nil

        if endSession {
            finishSession()
        } else {
            // 继续专注，新建 work 分段
            sessionManager.addSegment(label: .work)
            // 允许再次触发休息提醒
            restBannerDismissed = false
            showRestBanner = false
        }
    }

    // MARK: - Breathing Animation

    private func startBreathingAnimation() {
        withAnimation(
            .easeInOut(duration: 2.0)
            .repeatForever(autoreverses: true)
        ) {
            breatheScale = 1.05
        }
    }

    // MARK: - Helpers

    private var staminaRingColor: Color {
        switch staminaState {
        case .focused:    return .green
        case .fading:     return .orange
        case .depleted:   return .red
        case .recovering: return .blue
        }
    }

    private func formatElapsed(_ seconds: TimeInterval) -> String {
        let total = max(0, Int(seconds))
        let h = total / 3600
        let m = (total % 3600) / 60
        let s = total % 60
        if h > 0 {
            return String(format: "%02d:%02d:%02d", h, m, s)
        }
        return String(format: "%02d:%02d", m, s)
    }
}

#Preview {
    ActiveSessionView()
        .environmentObject(FluxService())
        .environmentObject(SessionManager())
        .environmentObject(BLEManager())
        .environmentObject(LiveActivityManager())
}
