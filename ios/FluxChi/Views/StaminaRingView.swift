import SwiftUI

struct StaminaRingView: View {
    let value: Double
    let state: StaminaState

    private var progress: Double { min(max(value / 100, 0), 1) }

    private var ringColor: Color {
        switch state {
        case .focused:    return .green
        case .fading:     return .orange
        case .depleted:   return .red
        case .recovering: return .blue
        }
    }

    var body: some View {
        ZStack {
            Circle()
                .fill(.ultraThinMaterial)
                .frame(width: 190, height: 190)

            Circle()
                .stroke(Color.primary.opacity(0.06), lineWidth: 10)
                .frame(width: 170, height: 170)

            Circle()
                .trim(from: 0, to: progress)
                .stroke(
                    AngularGradient(
                        colors: [ringColor.opacity(0.3), ringColor],
                        center: .center,
                        startAngle: .degrees(-90),
                        endAngle: .degrees(-90 + 360 * progress)
                    ),
                    style: StrokeStyle(lineWidth: 10, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
                .frame(width: 170, height: 170)
                .shadow(color: ringColor.opacity(0.35), radius: 10, y: 2)
                .animation(.spring(duration: 0.8), value: progress)

            VStack(spacing: 2) {
                Text("\(Int(value))")
                    .font(.system(size: 52, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)
                    .contentTransition(.numericText(value: value))
                    .animation(.easeInOut, value: Int(value))

                Text("STAMINA")
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .tracking(3)

                HStack(spacing: 4) {
                    Image(systemName: state.systemImage)
                        .font(.system(size: 11))
                    Text(state.displayName)
                        .font(.system(size: 12, weight: .semibold))
                }
                .foregroundStyle(ringColor)
                .padding(.top, 4)
            }
        }
        .frame(width: 200, height: 200)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("续航值 \(Int(value))%，状态 \(state.displayName)")
    }
}

#Preview {
    VStack(spacing: 40) {
        StaminaRingView(value: 78, state: .focused)
        StaminaRingView(value: 42, state: .fading)
        StaminaRingView(value: 15, state: .depleted)
    }
    .padding()
}
