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

    /// Deep link notification names
    static let showActiveSessionNotification = Notification.Name("FluxChi.showActiveSession")
    static let showRestFromNotification = Notification.Name("FluxChi.showRestFromNotification")

    init() {
        PerformanceMonitor.shared.markAppInit()
        configureLogger()
    }

    private func configureLogger() {
        #if DEBUG
        let config = FluxLogConfig.debug
        #else
        let config = FluxLogConfig.production
        #endif

        Task { @MainActor in
            FluxLogger.shared.updateConfig(config)
            FluxLog.app.info("FluxChi launch - v\(Flux.App.version)")
        }
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

                if #available(iOS 26.0, *) {
                    let engine = NLPSummaryEngine.shared
                    engine.prewarm()
                    FluxLog.app.info("NLP diagnostic: \(engine.diagnosticInfo)")
                }

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
            if sessionManager.isRecording {
                NotificationCenter.default.post(name: Self.showActiveSessionNotification, object: nil)
            }
        case "rest":
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

    // MARK: - Tab View + Focus Button

    @State private var showActiveSession = false
    @State private var showCalibrationAlert = false
    @State private var finishedSession: Session?
    @State private var showSummary = false
    @State private var selectedTab = "dashboard"
    @State private var showConnectionSheet = false

    private var isLive: Bool { service.isConnected || bleManager.isConnected }

    private var isCalibratedToday: Bool {
        let last = UserDefaults.standard.double(forKey: "flux_last_calibration")
        guard last > 0 else { return false }
        return Calendar.current.isDateInToday(Date(timeIntervalSince1970: last))
    }

    @ViewBuilder
    private var mainTabView: some View {
        nativeTabView
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
            .sheet(isPresented: $showConnectionSheet) {
                ConnectionGuideSheet()
            }
    }

    // MARK: - Native TabView + Tab(role: .search)

    @ViewBuilder
    private var nativeTabView: some View {
        if #available(iOS 18.0, *) {
            TabView(selection: $selectedTab) {
                Tab("仪表盘", systemImage: "waveform", value: "dashboard") {
                    DashboardView(showActiveSession: $showActiveSession, finishedSession: $finishedSession)
                }
                Tab("历史", systemImage: "clock.arrow.circlepath", value: "history") {
                    HistoryView()
                }
                Tab("设置", systemImage: "gearshape", value: "settings") {
                    SettingsView()
                }
                Tab(
                    "专注",
                    systemImage: sessionManager.isRecording
                        ? "brain.head.profile.fill"
                        : "brain.head.profile",
                    value: "focus",
                    role: .search
                ) {
                    Color.clear
                }
            }
            .tint(Flux.Colors.accent)
            .modifier(TabBarMinimizeModifier())
            .onChange(of: selectedTab) { _, newValue in
                guard newValue == "focus" else { return }
                var t = Transaction()
                t.disablesAnimations = true
                withTransaction(t) { selectedTab = "dashboard" }
                handleFocusTap()
            }
        } else {
            ZStack(alignment: .bottomTrailing) {
                TabView {
                    DashboardView(showActiveSession: $showActiveSession, finishedSession: $finishedSession)
                        .tabItem { Label("仪表盘", systemImage: "waveform") }
                    HistoryView()
                        .tabItem { Label("历史", systemImage: "clock.arrow.circlepath") }
                    SettingsView()
                        .tabItem { Label("设置", systemImage: "gearshape") }
                }
                .tint(Flux.Colors.accent)

                focusButton
                    .padding(.trailing, 20)
                    .padding(.bottom, 36)
            }
        }
    }

    private var focusButton: some View {
        Button { handleFocusTap() } label: {
            ZStack {
                Circle()
                    .fill(Flux.Colors.accent.gradient)
                    .frame(width: 52, height: 52)
                    .shadow(color: Flux.Colors.accent.opacity(0.4), radius: 10, y: 4)

                Image(systemName: sessionManager.isRecording ? "brain.head.profile.fill" : "brain.head.profile")
                    .font(.system(size: 22))
                    .foregroundStyle(.white)
            }
        }
        .buttonStyle(.plain)
    }

    private func handleFocusTap() {
        if sessionManager.isRecording {
            showActiveSession = true
        } else if !isLive {
            showConnectionSheet = true
        } else if !isCalibratedToday {
            showCalibrationAlert = true
        } else {
            beginFocusSession()
        }
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

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification
    ) async -> UNNotificationPresentationOptions {
        [.banner, .sound]
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse
    ) async {
        let userInfo = response.notification.request.content.userInfo

        if let action = userInfo["action"] as? String, action == "showActiveSession" {
            await MainActor.run {
                NotificationCenter.default.post(
                    name: FluxChiApp.showActiveSessionNotification,
                    object: nil
                )
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

// MARK: - TabBar Minimize (iOS 26+)

private struct TabBarMinimizeModifier: ViewModifier {
    func body(content: Content) -> some View {
        if #available(iOS 26.0, *) {
            content.tabBarMinimizeBehavior(.onScrollDown)
        } else {
            content
        }
    }
}
