import SwiftUI

struct StaminaRingView: View {
    /// `nil` 时显示「—」（无 EMG/无有效融合，与 Web 一致）
    let value: Double?
    let state: StaminaState
    var size: CGFloat = 200

    private var progress: Double {
        guard let value else { return 0 }
        return min(max(value / 100, 0), 1)
    }
    private var ringWidth: CGFloat { size * 0.05 }
    private var ringDiameter: CGFloat { size * 0.85 }
    private var bgDiameter: CGFloat { size * 0.95 }
    private var numberSize: CGFloat { size * 0.26 }

    private var ringColor: Color {
        Flux.Colors.forStaminaState(state)
    }

    var body: some View {
        ZStack {
            Circle()
                .fill(Color(.systemBackground).opacity(0.6))
                .frame(width: bgDiameter, height: bgDiameter)

            Circle()
                .stroke(Color.primary.opacity(0.06), lineWidth: ringWidth)
                .frame(width: ringDiameter, height: ringDiameter)

            Circle()
                .trim(from: 0, to: progress)
                .stroke(
                    AngularGradient(
                        colors: [ringColor.opacity(0.3), ringColor],
                        center: .center,
                        startAngle: .degrees(-90),
                        endAngle: .degrees(-90 + 360 * progress)
                    ),
                    style: StrokeStyle(lineWidth: ringWidth, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
                .frame(width: ringDiameter, height: ringDiameter)
                .shadow(color: ringColor.opacity(0.35), radius: 10, y: 2)
                .animation(.spring(duration: 0.8), value: progress)

            VStack(spacing: 2) {
                Text(value.map { "\(Int($0))" } ?? "—")
                    .font(.system(size: numberSize, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)
                    .contentTransition(.numericText(value: value ?? 0))
                    .animation(.easeInOut, value: value.map(Int.init) ?? -1)

                Text("STAMINA")
                    .font(.system(size: size * 0.045, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .tracking(3)

                HStack(spacing: 4) {
                    Image(systemName: state.systemImage)
                        .font(.system(size: size * 0.055))
                    Text(state.displayName)
                        .font(.system(size: size * 0.06, weight: .semibold))
                }
                .foregroundStyle(ringColor)
                .padding(.top, 4)
            }
        }
        .frame(width: size, height: size)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(value.map { "续航值 \(Int($0))%，状态 \(state.displayName)" } ?? "无续航读数，状态 \(state.displayName)")
    }
}

#Preview {
    VStack(spacing: 40) {
        StaminaRingView(value: 78.0, state: .focused)
        StaminaRingView(value: 42.0, state: .fading)
        StaminaRingView(value: 15.0, state: .depleted)
        StaminaRingView(value: nil, state: .focused)
    }
    .padding()
}
