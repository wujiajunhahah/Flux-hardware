import SwiftUI
import SwiftData

/// 在 `modelContainer` 已注入后尽早配置 `SessionManager`，避免仅依赖某个 Tab 的 `onAppear`。
struct SessionManagerBootstrapView: View {
    @Environment(\.modelContext) private var modelContext
    @EnvironmentObject private var sessionManager: SessionManager
    @EnvironmentObject private var service: FluxService

    var body: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .accessibilityHidden(true)
            .onAppear {
                sessionManager.configure(
                    modelContext: modelContext,
                    stateProvider: { [weak service] in service?.state }
                )
            }
    }
}
