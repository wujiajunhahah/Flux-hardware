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

    // MARK: - Tab View (iOS 18+ Tab API + iOS 26 minimize)

    @ViewBuilder
    private var mainTabView: some View {
        if #available(iOS 18.0, *) {
            TabView {
                Tab("仪表盘", systemImage: "waveform") {
                    DashboardView()
                }
                Tab("历史", systemImage: "clock.arrow.circlepath") {
                    HistoryView()
                }
                Tab("设置", systemImage: "gearshape") {
                    SettingsView()
                }
            }
            .tint(.red)
            .modifier(TabBarMinimizeModifier())
        } else {
            TabView {
                DashboardView()
                    .tabItem { Label("仪表盘", systemImage: "waveform") }
                HistoryView()
                    .tabItem { Label("历史", systemImage: "clock.arrow.circlepath") }
                SettingsView()
                    .tabItem { Label("设置", systemImage: "gearshape") }
            }
            .tint(.red)
        }
    }
}

// MARK: - iOS 26+ TabBar Minimize Modifier

private struct TabBarMinimizeModifier: ViewModifier {
    func body(content: Content) -> some View {
        if #available(iOS 26.0, *) {
            content.tabBarMinimizeBehavior(.onScrollDown)
        } else {
            content
        }
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
