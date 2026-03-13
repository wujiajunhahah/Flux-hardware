import SwiftUI
import SwiftData
import UserNotifications

@main
struct FluxChiApp: App {
    @StateObject private var service = FluxService()
    @StateObject private var bleManager = BLEManager()
    @StateObject private var sessionManager = SessionManager()
    @StateObject private var personalization = PersonalizationManager()
    @StateObject private var alertManager = AlertManager()
    @StateObject private var liveActivityManager = LiveActivityManager()

    @AppStorage("flux_onboarding_done") private var onboardingDone = false

    /// 深度链接通知名（点击灵动岛/通知 → 跳转 ActiveSessionView）
    static let showActiveSessionNotification = Notification.Name("FluxChi.showActiveSession")

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
        // fluxchi://session — 跳转到 ActiveSessionView
        guard url.scheme == "fluxchi", url.host == "session" else { return }
        if sessionManager.isRecording {
            NotificationCenter.default.post(name: Self.showActiveSessionNotification, object: nil)
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
