import ActivityKit
import WidgetKit
import SwiftUI

struct FluxChiLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: FluxChiLiveAttributes.self) { context in
            lockScreenBanner(context: context)
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("\(Int(context.state.stamina))")
                            .font(.system(size: 36, weight: .bold, design: .rounded))
                            .foregroundStyle(staminaColor(context.state.stamina))
                            .contentTransition(.numericText(value: context.state.stamina))

                        Text(stateLabel(context.state.state))
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }
                }

                DynamicIslandExpandedRegion(.trailing) {
                    VStack(alignment: .trailing, spacing: 6) {
                        HStack(spacing: 4) {
                            Image(systemName: "timer")
                                .font(.system(size: 10))
                            Text("\(Int(context.state.continuousWorkMin))m")
                                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                        }
                        .foregroundStyle(.secondary)

                        HStack(spacing: 3) {
                            dimensionDot(context.state.consistency, .blue)
                            dimensionDot(context.state.tension, .orange)
                            dimensionDot(context.state.fatigue, .red)
                        }
                    }
                }

                DynamicIslandExpandedRegion(.bottom) {
                    staminaBar(value: context.state.stamina)
                        .padding(.top, 4)
                }

                DynamicIslandExpandedRegion(.center) {}
            } compactLeading: {
                ZStack {
                    Circle()
                        .trim(from: 0, to: context.state.stamina / 100)
                        .stroke(staminaColor(context.state.stamina), style: StrokeStyle(lineWidth: 3, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                        .frame(width: 22, height: 22)

                    Text("\(Int(context.state.stamina))")
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                }
            } compactTrailing: {
                Image(systemName: stateIcon(context.state.state))
                    .font(.system(size: 12))
                    .foregroundStyle(staminaColor(context.state.stamina))
            } minimal: {
                ZStack {
                    Circle()
                        .trim(from: 0, to: context.state.stamina / 100)
                        .stroke(staminaColor(context.state.stamina), lineWidth: 2)
                        .rotationEffect(.degrees(-90))

                    Text("\(Int(context.state.stamina))")
                        .font(.system(size: 9, weight: .bold, design: .rounded))
                }
            }
        }
    }

    // MARK: - Lock Screen Banner

    @ViewBuilder
    private func lockScreenBanner(context: ActivityViewContext<FluxChiLiveAttributes>) -> some View {
        HStack(spacing: 16) {
            ZStack {
                Circle()
                    .stroke(Color.secondary.opacity(0.2), lineWidth: 4)
                    .frame(width: 50, height: 50)
                Circle()
                    .trim(from: 0, to: context.state.stamina / 100)
                    .stroke(staminaColor(context.state.stamina),
                            style: StrokeStyle(lineWidth: 4, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .frame(width: 50, height: 50)

                VStack(spacing: 0) {
                    Text("\(Int(context.state.stamina))")
                        .font(.system(size: 18, weight: .bold, design: .rounded))
                    Text("Stamina")
                        .font(.system(size: 7, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(context.attributes.sessionTitle)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)

                HStack(spacing: 12) {
                    Label(stateLabel(context.state.state),
                          systemImage: stateIcon(context.state.state))
                        .foregroundStyle(staminaColor(context.state.stamina))

                    Label("\(Int(context.state.continuousWorkMin))m",
                          systemImage: "timer")
                        .foregroundStyle(.secondary)
                }
                .font(.caption)
            }

            Spacer()

            VStack(spacing: 3) {
                dimensionMini("C", context.state.consistency, .blue)
                dimensionMini("T", context.state.tension, .orange)
                dimensionMini("F", context.state.fatigue, .red)
            }
        }
        .padding(16)
        .background(.ultraThinMaterial)
    }

    // MARK: - Components

    private func staminaBar(value: Double) -> some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 3)
                    .fill(Color.secondary.opacity(0.15))

                RoundedRectangle(cornerRadius: 3)
                    .fill(staminaColor(value).gradient)
                    .frame(width: max(4, geo.size.width * value / 100))
            }
        }
        .frame(height: 6)
    }

    private func dimensionDot(_ value: Double, _ color: Color) -> some View {
        Circle()
            .fill(color.opacity(0.3 + value * 0.7))
            .frame(width: 8 + value * 6, height: 8 + value * 6)
    }

    private func dimensionMini(_ letter: String, _ value: Double, _ color: Color) -> some View {
        HStack(spacing: 3) {
            Text(letter)
                .font(.system(size: 8, weight: .bold, design: .monospaced))
                .foregroundStyle(.secondary)
            RoundedRectangle(cornerRadius: 1.5)
                .fill(color.opacity(0.3 + value * 0.7))
                .frame(width: 20 * value + 4, height: 4)
        }
    }

    private func staminaColor(_ value: Double) -> Color {
        if value > 60 { return .green }
        if value > 30 { return .orange }
        return .red
    }

    private func stateLabel(_ state: String) -> String {
        switch state {
        case "focused":    return "专注"
        case "fading":     return "下降"
        case "depleted":   return "耗尽"
        case "recovering": return "恢复"
        default:           return state
        }
    }

    private func stateIcon(_ state: String) -> String {
        switch state {
        case "focused":    return "bolt.fill"
        case "fading":     return "bolt.trianglebadge.exclamationmark"
        case "depleted":   return "bolt.slash.fill"
        case "recovering": return "leaf.fill"
        default:           return "circle"
        }
    }
}
