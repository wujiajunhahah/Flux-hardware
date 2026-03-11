import ActivityKit
import WidgetKit
import SwiftUI

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
                    HStack(alignment: .center, spacing: 10) {
                        LiveMiniRing(progress: progress, color: color, size: 36)

                        VStack(alignment: .leading, spacing: 1) {
                            Text(stateLabel(context.state.state))
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(color)

                            Text("Stamina")
                                .font(.system(size: 9, weight: .regular))
                                .foregroundStyle(.white.opacity(0.4))
                                .textCase(.uppercase)
                                .tracking(0.5)
                        }
                    }
                }

                DynamicIslandExpandedRegion(.trailing) {
                    VStack(alignment: .trailing, spacing: 4) {
                        dimensionRow("C", context.state.consistency, .cyan)
                        dimensionRow("T", context.state.tension, .orange)
                        dimensionRow("F", context.state.fatigue, .pink)
                    }
                    .padding(.top, 4)
                }

                DynamicIslandExpandedRegion(.bottom) {
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule()
                                .fill(.white.opacity(0.08))

                            Capsule()
                                .fill(
                                    LinearGradient(
                                        colors: [color.opacity(0.6), color],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .frame(width: max(4, geo.size.width * progress))
                                .shadow(color: color.opacity(0.5), radius: 6, y: 0)
                        }
                    }
                    .frame(height: 3)
                    .padding(.top, 6)
                }

                DynamicIslandExpandedRegion(.center) {}

            } compactLeading: {
                LiveMiniRing(progress: progress, color: color, size: 22)

            } compactTrailing: {
                Image(systemName: stateIcon(context.state.state))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(color)
                    .symbolRenderingMode(.hierarchical)

            } minimal: {
                LiveMiniRing(progress: progress, color: color, size: 16)
            }
        }
    }

    // MARK: - Lock Screen Banner

    @ViewBuilder
    private func lockScreenBanner(context: ActivityViewContext<FluxChiLiveAttributes>) -> some View {
        let color = staminaColor(context.state.state)
        let progress = clampProgress(context.state.stamina)

        HStack(spacing: 16) {
            // Stamina ring
            LiveStaminaRing(value: context.state.stamina, color: color, size: 50)

            // Primary info
            VStack(alignment: .leading, spacing: 4) {
                Text(context.attributes.sessionTitle)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                HStack(spacing: 4) {
                    Image(systemName: stateIcon(context.state.state))
                        .font(.system(size: 10))
                    Text(stateLabel(context.state.state))
                        .font(.caption.weight(.medium))
                }
                .foregroundStyle(color)
            }

            Spacer()

            // Three dimension vertical bars
            HStack(spacing: 5) {
                dimensionPill(context.state.consistency, .cyan)
                dimensionPill(context.state.tension, .orange)
                dimensionPill(context.state.fatigue, .pink)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
        .background {
            ZStack {
                // Subtle gradient backdrop
                LinearGradient(
                    colors: [
                        color.opacity(0.08),
                        Color(.systemBackground).opacity(0.02)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )

                // Glass material layer
                Rectangle().fill(.ultraThinMaterial)
            }
        }
    }

    // MARK: - Expanded: dimension row

    private func dimensionRow(_ letter: String, _ value: Double, _ color: Color) -> some View {
        HStack(spacing: 4) {
            Text(letter)
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .foregroundStyle(.white.opacity(0.35))

            ZStack(alignment: .leading) {
                Capsule()
                    .fill(.white.opacity(0.08))
                    .frame(width: 28, height: 3)

                Capsule()
                    .fill(color)
                    .frame(width: max(2, 28 * min(value, 1)), height: 3)
            }
        }
    }

    // MARK: - Lock screen: dimension pill

    private func dimensionPill(_ value: Double, _ color: Color) -> some View {
        let height: CGFloat = 28
        let fillHeight = max(3, height * min(value, 1))

        return ZStack(alignment: .bottom) {
            RoundedRectangle(cornerRadius: 2.5)
                .fill(color.opacity(0.1))
                .frame(width: 5, height: height)

            RoundedRectangle(cornerRadius: 2.5)
                .fill(
                    LinearGradient(
                        colors: [color.opacity(0.5), color],
                        startPoint: .bottom,
                        endPoint: .top
                    )
                )
                .frame(width: 5, height: fillHeight)
        }
    }

    // MARK: - Helpers

    private func clampProgress(_ stamina: Double) -> Double {
        min(max(stamina / 100, 0), 1)
    }

    private func staminaColor(_ state: String) -> Color {
        switch state {
        case "focused":    return Color(red: 0.30, green: 0.85, blue: 0.50)
        case "fading":     return Color(red: 1.00, green: 0.70, blue: 0.20)
        case "depleted":   return Color(red: 1.00, green: 0.35, blue: 0.30)
        case "recovering": return Color(red: 0.25, green: 0.60, blue: 1.00)
        default:           return .gray
        }
    }

    private func stateLabel(_ state: String) -> String {
        switch state {
        case "focused":    return "Focused"
        case "fading":     return "Fading"
        case "depleted":   return "Depleted"
        case "recovering": return "Recovering"
        default:           return state
        }
    }

    private func stateIcon(_ state: String) -> String {
        switch state {
        case "focused":    return "bolt.fill"
        case "fading":     return "bolt.badge.clock.fill"
        case "depleted":   return "bolt.slash.fill"
        case "recovering": return "leaf.fill"
        default:           return "circle"
        }
    }
}

// MARK: - LiveMiniRing

/// Compact ring for Dynamic Island (dark background context).
/// Uses gradient stroke with a subtle glow — no number inside at small sizes.
private struct LiveMiniRing: View {
    let progress: Double
    let color: Color
    let size: CGFloat

    private var lineWidth: CGFloat { max(2, size * 0.12) }
    private var showNumber: Bool { size >= 20 }

    var body: some View {
        ZStack {
            Circle()
                .stroke(.white.opacity(0.08), lineWidth: lineWidth)

            Circle()
                .trim(from: 0, to: progress)
                .stroke(
                    AngularGradient(
                        colors: [color.opacity(0.3), color],
                        center: .center,
                        startAngle: .degrees(-90),
                        endAngle: .degrees(-90 + 360 * progress)
                    ),
                    style: StrokeStyle(lineWidth: lineWidth, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
                .shadow(color: color.opacity(0.4), radius: 4, y: 0)

            if showNumber {
                Text("\(Int(progress * 100))")
                    .font(.system(size: size * 0.3, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                    .contentTransition(.numericText(value: progress))
            }
        }
        .frame(width: size, height: size)
    }
}

// MARK: - LiveStaminaRing

/// Larger ring for lock screen banner (light/dark adaptive).
private struct LiveStaminaRing: View {
    let value: Double
    let color: Color
    let size: CGFloat

    private var progress: Double { min(max(value / 100, 0), 1) }
    private var lineWidth: CGFloat { size * 0.08 }

    var body: some View {
        ZStack {
            Circle()
                .stroke(color.opacity(0.1), lineWidth: lineWidth)

            Circle()
                .trim(from: 0, to: progress)
                .stroke(
                    AngularGradient(
                        colors: [color.opacity(0.35), color],
                        center: .center,
                        startAngle: .degrees(-90),
                        endAngle: .degrees(-90 + 360 * progress)
                    ),
                    style: StrokeStyle(lineWidth: lineWidth, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))

            VStack(spacing: 0) {
                Text("\(Int(value))")
                    .font(.system(size: size * 0.28, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)
                    .contentTransition(.numericText(value: value))
            }
        }
        .frame(width: size, height: size)
    }
}
