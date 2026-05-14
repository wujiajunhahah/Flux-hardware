import ActivityKit
import WidgetKit
import SwiftUI

// MARK: - Brand tokens (extension-target local)
//
// FluxChi 主 app 的 Flux.Colors / FluxTokens 没编入 FluxChiLive target（避免
// 把整个 Domain 依赖拖进 widget extension）。这里复刻 ShareCardContent 的关键
// 视觉常量：暗底 + accent 红橙 + 大圆角 + mono tracking。

private extension Color {
    /// FocuX 主色 — 与 ShareCard / Dashboard / StaminaRing 同源
    static let focuxAccent = Color(red: 0.90, green: 0.20, blue: 0.20)
    /// LockScreen 渐变深色端
    static let focuxDeepBg = Color(red: 0.02, green: 0.02, blue: 0.04)
}

/// Live Activity 主体 — 同时管 Lock Screen banner 与 Dynamic Island。
///
/// 设计参考 `ShareCardContent`（InsightShareCard.swift）：暗渐变背景 + sparkles
/// + focux wordmark + 大数字 + 跳秒计时。跨设备视感统一。
struct FluxChiLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: FluxChiLiveAttributes.self) { context in
            lockScreenBanner(context: context)
        } dynamicIsland: { context in
            let color = staminaColor(context.state.state)
            let progress = clampProgress(context.state.stamina)

            return DynamicIsland {
                // MARK: Expanded

                DynamicIslandExpandedRegion(.leading) {
                    HStack(alignment: .center, spacing: 12) {
                        LiveRing(progress: progress, color: color, size: 44)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(stateLabel(context.state.state))
                                .font(.system(size: 15, weight: .semibold, design: .rounded))
                                .foregroundStyle(color)
                                .lineLimit(1)

                            Text("Stamina \(Int(context.state.stamina))")
                                .font(.system(size: 10, weight: .medium, design: .monospaced))
                                .foregroundStyle(.white.opacity(0.5))
                                .tracking(0.3)
                        }
                    }
                }

                DynamicIslandExpandedRegion(.trailing) {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text(timerInterval: context.attributes.startedAt...Date.distantFuture,
                             countsDown: false)
                            .font(.system(size: 22, weight: .semibold, design: .rounded))
                            .monospacedDigit()
                            .foregroundStyle(.white)
                            .multilineTextAlignment(.trailing)
                            .lineLimit(1)
                            .frame(maxWidth: 90, alignment: .trailing)

                        Text("已专注")
                            .font(.system(size: 9, weight: .medium))
                            .foregroundStyle(.white.opacity(0.5))
                            .textCase(.uppercase)
                            .tracking(0.6)
                    }
                }

                DynamicIslandExpandedRegion(.bottom) {
                    // 细 stamina progress bar，与 ShareCard 一致的渐变填充
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule()
                                .fill(.white.opacity(0.08))

                            Capsule()
                                .fill(
                                    LinearGradient(
                                        colors: [color.opacity(0.55), color],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .frame(width: max(4, geo.size.width * progress))
                        }
                    }
                    .frame(height: 3)
                    .padding(.top, 8)
                }

            } compactLeading: {
                LiveRing(progress: progress, color: color, size: 20)

            } compactTrailing: {
                // 灵动岛 compact 右侧显示「已专注分钟数」而不是状态 icon — 信息密度更高
                Text(timerInterval: context.attributes.startedAt...Date.distantFuture,
                     countsDown: false,
                     showsHours: false)
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.white)
                    .monospacedDigit()
                    .frame(maxWidth: 48, alignment: .trailing)
                    .lineLimit(1)

            } minimal: {
                LiveRing(progress: progress, color: color, size: 18)
            }
        }
    }

    // MARK: - Lock Screen Banner (share-card aesthetic)
    //
    // 关键设计选择：
    //   - 深色渐变 (black → accent 0.55) 作为背景 → 跟 ShareCard 一致
    //   - sparkles + focux wordmark 在左上 → 品牌识别
    //   - 大续航数字 + 跳秒计时器 → 信息密度最高
    //   - Lock Screen 强制深色看起来好；浅色模式系统会自动渲染对比

    @ViewBuilder
    private func lockScreenBanner(context: ActivityViewContext<FluxChiLiveAttributes>) -> some View {
        let color = staminaColor(context.state.state)
        let progress = clampProgress(context.state.stamina)

        ZStack {
            // 深色渐变背景
            LinearGradient(
                colors: [Color.focuxDeepBg, Color.focuxAccent.opacity(0.55)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            VStack(alignment: .leading, spacing: 12) {
                // Header: sparkles + wordmark
                HStack(spacing: 6) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 11, weight: .semibold))
                    Text("focux")
                        .font(.system(size: 11, weight: .semibold))
                        .tracking(0.8)
                    Spacer()
                    // Right corner: state pill
                    Text(stateLabel(context.state.state))
                        .font(.system(size: 11, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(color.opacity(0.85), in: Capsule())
                }
                .foregroundStyle(.white.opacity(0.85))

                // Main row: stamina ring + elapsed time
                HStack(alignment: .center, spacing: 18) {
                    LiveRing(progress: progress, color: .white, size: 64, lineWidth: 5, showNumber: true)

                    Spacer(minLength: 0)

                    VStack(alignment: .trailing, spacing: 2) {
                        Text(timerInterval: context.attributes.startedAt...Date.distantFuture,
                             countsDown: false)
                            .font(.system(size: 36, weight: .bold, design: .rounded))
                            .monospacedDigit()
                            .foregroundStyle(.white)
                            .lineLimit(1)
                            .minimumScaleFactor(0.7)

                        Text("已专注")
                            .font(.system(size: 10, weight: .medium))
                            .tracking(0.6)
                            .textCase(.uppercase)
                            .foregroundStyle(.white.opacity(0.65))
                    }
                }

                // 细进度条 — 跟 ShareCard 视觉一致
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule()
                            .fill(.white.opacity(0.15))
                        Capsule()
                            .fill(.white)
                            .frame(width: max(4, geo.size.width * progress))
                    }
                }
                .frame(height: 3)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 16)
        }
    }

    // MARK: - Helpers

    private func clampProgress(_ stamina: Double) -> Double {
        min(max(stamina / 100, 0), 1)
    }

    /// 主 app `Flux.Colors.forStaminaState` 同源（StaminaStatePalette 是双 target 共享）。
    private func staminaColor(_ state: String) -> Color {
        StaminaStatePalette.color(forRawState: state)
    }

    /// 中文状态标签 — 与主 app InsightStats / DashboardView 文案保持一致风格。
    private func stateLabel(_ state: String) -> String {
        switch state {
        case "focused":    return "专注稳定"
        case "fading":     return "节奏在调"
        case "depleted":   return "需要休息"
        case "recovering": return "恢复中"
        default:           return state
        }
    }
}

// MARK: - LiveRing
//
// 统一的环形组件 — Dynamic Island 与 Lock Screen 共用。
// 暗背景渲染优化：默认细线 + 数字在 ≥22pt 时显示。

private struct LiveRing: View {
    let progress: Double
    let color: Color
    let size: CGFloat
    var lineWidth: CGFloat? = nil
    var showNumber: Bool? = nil

    private var resolvedLineWidth: CGFloat {
        lineWidth ?? max(2, size * 0.12)
    }
    private var shouldShowNumber: Bool {
        showNumber ?? (size >= 22)
    }

    var body: some View {
        ZStack {
            Circle()
                .stroke(color.opacity(0.18), lineWidth: resolvedLineWidth)

            Circle()
                .trim(from: 0, to: progress)
                .stroke(
                    color,
                    style: StrokeStyle(lineWidth: resolvedLineWidth, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))

            if shouldShowNumber {
                Text("\(Int(progress * 100))")
                    .font(.system(size: size * 0.34, weight: .bold, design: .rounded))
                    .foregroundStyle(color)
                    .contentTransition(.numericText(value: progress))
            }
        }
        .frame(width: size, height: size)
    }
}
