import SwiftUI
import SwiftData

@main
struct FluxChiApp: App {
    @StateObject private var service = FluxService()
    @StateObject private var bleManager = BLEManager()
    @StateObject private var sessionManager = SessionManager()
    @StateObject private var personalization = PersonalizationManager()

    var body: some Scene {
        WindowGroup {
            TabView {
                DashboardView()
                    .tabItem {
                        Label("状态", systemImage: "gauge.with.dots.needle.67percent")
                    }

                RecorderView()
                    .tabItem {
                        Label("记录", systemImage: "record.circle")
                    }

                HistoryView()
                    .tabItem {
                        Label("历史", systemImage: "clock.arrow.circlepath")
                    }

                SettingsView()
                    .tabItem {
                        Label("设置", systemImage: "gearshape")
                    }
            }
            .tint(.red)
            .environmentObject(service)
            .environmentObject(bleManager)
            .environmentObject(sessionManager)
            .environmentObject(personalization)
            .onAppear {
                service.startPolling()
                bleManager.onStateUpdate = { [weak service] state in
                    Task { @MainActor in
                        service?.state = state
                        service?.isConnected = true
                        service?.connectionError = nil
                    }
                }
            }
            .onChange(of: bleManager.isConnected) { _, connected in
                if connected {
                    service.stopPolling()
                } else {
                    service.startPolling()
                }
            }
        }
        .modelContainer(for: [
            Session.self,
            Segment.self,
            FluxSnapshot.self,
            UserFeedback.self
        ])
    }
}
