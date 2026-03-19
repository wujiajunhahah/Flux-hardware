import SwiftUI
import SwiftData

@main
struct FluxChiApp: App {
    @StateObject private var service = FluxService()
    @StateObject private var bleManager = BLEManager()
    @StateObject private var sessionManager = SessionManager()
    @StateObject private var personalization = PersonalizationManager()
    @StateObject private var alertManager = AlertManager()
    @StateObject private var liveActivityManager = LiveActivityManager()

    // MARK: - Logger Configuration

    init() {
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
            FluxLogger.shared.info("FluxChi 启动 - v\(Flux.App.version)", category: .app)
        }
    }

    var body: some Scene {
        WindowGroup {
            mainTabView
            .environmentObject(service)
            .environmentObject(bleManager)
            .environmentObject(sessionManager)
            .environmentObject(personalization)
            .environmentObject(alertManager)
            .environmentObject(liveActivityManager)
            .onAppear {
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
        }
        .modelContainer(for: [
            Session.self, Segment.self, FluxSnapshot.self, UserFeedback.self
        ])
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
