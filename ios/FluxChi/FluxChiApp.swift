import SwiftUI

@main
struct FluxChiApp: App {
    @StateObject private var service = FluxService()
    @StateObject private var bleManager = BLEManager()

    var body: some Scene {
        WindowGroup {
            TabView {
                DashboardView()
                    .tabItem {
                        Label("仪表盘", systemImage: "gauge.with.dots.needle.67percent")
                    }

                BLEView()
                    .tabItem {
                        Label("蓝牙", systemImage: "antenna.radiowaves.left.and.right")
                    }

                SettingsView()
                    .tabItem {
                        Label("设置", systemImage: "gearshape")
                    }
            }
            .tint(.red)
            .environmentObject(service)
            .environmentObject(bleManager)
            .onAppear {
                service.startSSE()
            }
            .onDisappear {
                service.stopSSE()
            }
        }
    }
}
