import SwiftUI

struct StaminaRingView: View {
    let value: Double      // 0–100
    let state: StaminaState

    private var progress: Double { min(max(value / 100, 0), 1) }

    private var ringColor: Color {
        switch state {
        case .focused:    return .red
        case .fading:     return .orange
        case .depleted:   return .red.opacity(0.6)
        case .recovering: return .green
        }
    }

    var body: some View {
        ZStack {
            // Track
            Circle()
                .stroke(Color(.systemGray5), lineWidth: 6)

            // Dot pattern ring
            Circle()
                .stroke(Color(.systemGray4), style: StrokeStyle(lineWidth: 2, dash: [2, 6]))
                .padding(-8)

            // Progress arc
            Circle()
                .trim(from: 0, to: progress)
                .stroke(
                    ringColor,
                    style: StrokeStyle(lineWidth: 8, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
                .shadow(color: ringColor.opacity(0.4), radius: 8)
                .animation(.easeInOut(duration: 0.8), value: progress)

            // Center content
            VStack(spacing: 4) {
                Text("\(Int(value))")
                    .font(.system(size: 56, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)
                    .contentTransition(.numericText(value: value))
                    .animation(.easeInOut, value: Int(value))

                Text("STAMINA")
                    .font(.caption2)
                    .fontWeight(.medium)
                    .foregroundStyle(.secondary)
                    .tracking(2)

                Label(state.displayName, systemImage: state.systemImage)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(ringColor)
                    .padding(.top, 2)
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
