import SwiftUI
import SwiftData

/// 在 `modelContainer` 已注入后尽早配置 `SessionManager`，避免仅依赖某个 Tab 的 `onAppear`。
///
/// **放置约束**：须挂在 `WindowGroup` 内 `Group` 的**最前**、且在 `if onboardingDone` 等分支**之上**，
/// 保证首帧就会创建本视图，从而 `onAppear` 能及时执行；若挪到某 Tab 内，configure 可能延迟到该 Tab 首次可见。
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
