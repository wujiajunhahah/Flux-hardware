import SwiftUI
import SwiftData
import UserNotifications

@main
struct FluxChiApp: App {
    @UIApplicationDelegateAdaptor(FluxChiAppDelegate.self) var appDelegate

    @StateObject private var service = FluxService()
    @StateObject private var bleManager = BLEManager()
    @StateObject private var sessionManager = SessionManager()
    @StateObject private var personalization = PersonalizationManager()
    @StateObject private var alertManager = AlertManager()
    @StateObject private var liveActivityManager = LiveActivityManager()

    @AppStorage("flux_onboarding_done") private var onboardingDone = false

    /// 深度链接通知名（点击灵动岛/通知 → 跳转 ActiveSessionView）
    static let showActiveSessionNotification = Notification.Name("FluxChi.showActiveSession")
    /// 通知点击触发休息 Banner
    static let showRestFromNotification = Notification.Name("FluxChi.showRestFromNotification")

    init() {
        PerformanceMonitor.shared.markAppInit()
    }

    var body: some Scene {
        WindowGroup {
            Group {
                if onboardingDone {
                    mainTabView
                } else {
                    OnboardingView(isCompleted: $onboardingDone)
                }
            }
            .environmentObject(service)
            .environmentObject(bleManager)
            .environmentObject(sessionManager)
            .environmentObject(personalization)
            .environmentObject(alertManager)
            .environmentObject(liveActivityManager)
            .onAppear {
                PerformanceMonitor.shared.markFirstFrame()
                service.personalization = personalization
                service.startPolling()
                alertManager.requestPermission()

                bleManager.onStateUpdate = { [weak service, weak alertManager, weak liveActivityManager] state in
                    Task { @MainActor in
                        service?.state = state
                        service?.isConnected = true
                        service?.connectionError = nil

                        if let s = state.stamina {
                            alertManager?.evaluate(
                                stamina: s.value,
                                state: s.state,
                                continuousWorkMin: s.continuousWorkMin
                            )

                            liveActivityManager?.updateActivity(
                                stamina: s.value,
                                state: s.state,
                                activity: state.activity,
                                consistency: s.consistency,
                                tension: s.tension,
                                fatigue: s.fatigue
                            )
                        }
                    }
                }
            }
            .onChange(of: bleManager.isConnected) { _, connected in
                if connected { service.stopPolling() } else { service.startPolling() }
            }
            .alert(alertManager.alertTitle, isPresented: $alertManager.showBreakAlert) {
                Button("休息一下") { alertManager.dismissAlert() }
                Button("继续工作", role: .cancel) { alertManager.dismissAlert() }
            } message: {
                Text(alertManager.alertMessage)
            }
            .onOpenURL { url in
                handleDeepLink(url)
            }
        }
        .modelContainer(for: [
            Session.self, Segment.self, FluxSnapshot.self, UserFeedback.self
        ])
    }

    // MARK: - Deep Link

    private func handleDeepLink(_ url: URL) {
        guard url.scheme == "fluxchi" else { return }

        switch url.host {
        case "session":
            // fluxchi://session — 跳转到 ActiveSessionView
            if sessionManager.isRecording {
                NotificationCenter.default.post(name: Self.showActiveSessionNotification, object: nil)
            }
        case "rest":
            // fluxchi://rest — 灵动岛「一键休息」→ 跳转 + 触发休息 Banner
            if sessionManager.isRecording {
                NotificationCenter.default.post(name: Self.showActiveSessionNotification, object: nil)
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    NotificationCenter.default.post(name: Self.showRestFromNotification, object: nil)
                }
            }
        default:
            break
        }
    }

    // MARK: - Custom Tab View with Center Focus Button

    enum AppTab: Int { case dashboard, history, settings }

    @State private var selectedTab: AppTab = .dashboard
    @State private var showActiveSession = false
    @State private var showCalibrationAlert = false
    @State private var finishedSession: Session?
    @State private var showSummary = false

    private var isLive: Bool { service.isConnected || bleManager.isConnected }

    private var isCalibratedToday: Bool {
        let last = UserDefaults.standard.double(forKey: "flux_last_calibration")
        guard last > 0 else { return false }
        return Calendar.current.isDateInToday(Date(timeIntervalSince1970: last))
    }

    @ViewBuilder
    private var mainTabView: some View {
        ZStack(alignment: .bottom) {
            // 内容区
            Group {
                switch selectedTab {
                case .dashboard: DashboardView(showActiveSession: $showActiveSession, finishedSession: $finishedSession)
                case .history:   HistoryView()
                case .settings:  SettingsView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // 自定义 TabBar
            customTabBar
        }
        .fullScreenCover(isPresented: $showActiveSession) {
            ActiveSessionView { completedSession in
                finishedSession = completedSession
            }
        }
        .onChange(of: showActiveSession) { _, isShowing in
            if !isShowing, finishedSession != nil {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                    showSummary = true
                }
            }
        }
        .sheet(isPresented: $showSummary) {
            if let s = finishedSession { SessionSummarySheet(session: s) }
        }
        .alert("今日未校准", isPresented: $showCalibrationAlert) {
            Button("跳过，直接开始") { beginFocusSession() }
            Button("取消", role: .cancel) {}
        } message: {
            Text("校准可提高续航值的准确度。建议每天首次专注前完成校准。")
        }
    }

    private var customTabBar: some View {
        HStack(spacing: 0) {
            // 左：仪表盘
            tabBarItem(icon: "waveform", label: "仪表盘", tab: .dashboard)
                .frame(maxWidth: .infinity)

            // 左中：历史
            tabBarItem(icon: "clock.arrow.circlepath", label: "历史", tab: .history)
                .frame(maxWidth: .infinity)

            // 中心：圆形专注按钮
            focusButton
                .offset(y: -22)
                .frame(maxWidth: .infinity)

            // 右中：设置
            tabBarItem(icon: "gearshape", label: "设置", tab: .settings)
                .frame(maxWidth: .infinity)
        }
        .padding(.top, 10)
        .padding(.bottom, 6)
        .background(
            Rectangle()
                .fill(.ultraThinMaterial)
                .ignoresSafeArea(edges: .bottom)
                .shadow(color: .black.opacity(0.06), radius: 8, y: -2)
        )
    }

    private func tabBarItem(icon: String, label: String, tab: AppTab) -> some View {
        Button {
            withAnimation(.spring(duration: 0.25)) { selectedTab = tab }
        } label: {
            VStack(spacing: 3) {
                Image(systemName: icon)
                    .font(.system(size: 20))
                Text(label)
                    .font(.system(size: 10))
            }
            .foregroundStyle(selectedTab == tab ? Flux.Colors.accent : .secondary)
        }
        .buttonStyle(.plain)
    }

    private var focusButton: some View {
        Button {
            if sessionManager.isRecording {
                showActiveSession = true
            } else if !isCalibratedToday {
                showCalibrationAlert = true
            } else {
                beginFocusSession()
            }
        } label: {
            ZStack {
                Circle()
                    .fill(Flux.Colors.accent.gradient)
                    .frame(width: 60, height: 60)
                    .shadow(color: Flux.Colors.accent.opacity(0.4), radius: 10, y: 4)

                Image(systemName: sessionManager.isRecording ? "brain.head.profile.fill" : "brain.head.profile")
                    .font(.system(size: 24))
                    .foregroundStyle(.white)
            }
        }
        .buttonStyle(.plain)
        .disabled(!isLive && !sessionManager.isRecording)
        .opacity(isLive || sessionManager.isRecording ? 1.0 : 0.5)
    }

    private func beginFocusSession() {
        let source: SessionSource = bleManager.isConnected ? .ble : .wifi
        sessionManager.startSession(source: source)

        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "zh_CN")
        fmt.dateFormat = "HH:mm"
        liveActivityManager.startActivity(title: "专注中 · \(fmt.string(from: Date()))")

        showActiveSession = true
    }
}

// MARK: - App Delegate (UNUserNotificationCenterDelegate)

final class FluxChiAppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate {

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        UNUserNotificationCenter.current().delegate = self
        return true
    }

    /// 前台收到通知 → 仍然显示 banner
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        [.banner, .sound]
    }

    /// 用户点击通知 → 跳转 ActiveSessionView + 触发休息 Banner
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse
    ) async {
        let userInfo = response.notification.request.content.userInfo

        if let action = userInfo["action"] as? String, action == "showActiveSession" {
            await MainActor.run {
                // 先跳转到 ActiveSessionView
                NotificationCenter.default.post(
                    name: FluxChiApp.showActiveSessionNotification,
                    object: nil
                )
                // 如果携带 showRest 标记，触发休息 Banner
                if let showRest = userInfo["showRest"] as? Bool, showRest {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        NotificationCenter.default.post(
                            name: FluxChiApp.showRestFromNotification,
                            object: nil
                        )
                    }
                }
            }
        }
    }
}
