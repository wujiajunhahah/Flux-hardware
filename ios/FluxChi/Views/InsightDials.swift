import SwiftUI

// MARK: - StaminaDial

/// Canvas 绘制的环形仪表盘，支持 conicGradient + 数字居中。
struct StaminaDial: View {
    let value: Double  // 0...100
    let tint: Color

    private var progress: Double { max(0, min(1, value / 100)) }

    var body: some View {
        Canvas { context, size in
            let lineWidth: CGFloat = 9
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let radius = min(size.width, size.height) / 2 - lineWidth / 2 - 2

            // Background track
            let bg = Path { p in
                p.addArc(
                    center: center,
                    radius: radius,
                    startAngle: .degrees(-90),
                    endAngle: .degrees(270),
                    clockwise: false
                )
            }
            context.stroke(bg, with: .color(tint.opacity(0.12)),
                           style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))

            // Progress
            if progress > 0 {
                let endAngle = -90.0 + 360.0 * progress
                let fg = Path { p in
                    p.addArc(
                        center: center,
                        radius: radius,
                        startAngle: .degrees(-90),
                        endAngle: .degrees(endAngle),
                        clockwise: false
                    )
                }
                let gradient = Gradient(colors: [tint.opacity(0.75), tint])
                context.stroke(
                    fg,
                    with: .conicGradient(
                        gradient,
                        center: center,
                        angle: .degrees(-90)
                    ),
                    style: StrokeStyle(lineWidth: lineWidth, lineCap: .round)
                )
            }
        }
        .overlay {
            VStack(spacing: 0) {
                Text("\(Int(value))")
                    .font(.system(size: 32, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)
                    .contentTransition(.numericText(value: value))
                Text("续航")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.tertiary)
                    .padding(.top, -2)
            }
        }
        .animation(.spring(response: 0.6, dampingFraction: 0.85), value: value)
    }
}

// MARK: - TrendBadge

/// 上下箭头 + 数值 capsule，标注 delta 的方向。
struct TrendBadge: View {
    let delta: Int

    private var up: Bool { delta >= 0 }
    private var absText: String { delta > 0 ? "+\(delta)" : "\(delta)" }

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: up ? "arrow.up.right" : "arrow.down.right")
                .font(.system(size: 9, weight: .bold))
            Text(absText)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
        }
        .foregroundStyle(up ? Flux.Colors.success : Flux.Colors.warning)
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(
            (up ? Flux.Colors.success : Flux.Colors.warning).opacity(0.12),
            in: Capsule()
        )
    }
}

// MARK: - DotIndicator

/// 页面切换指示点，当前页为细长 capsule。
struct DotIndicator: View {
    let count: Int
    let selected: Int

    var body: some View {
        HStack(spacing: 6) {
            ForEach(0..<count, id: \.self) { idx in
                Capsule()
                    .fill(idx == selected ? Color.primary.opacity(0.8) : Color.primary.opacity(0.15))
                    .frame(width: idx == selected ? 18 : 6, height: 6)
                    .animation(.spring(response: 0.4, dampingFraction: 0.8), value: selected)
            }
        }
    }
}
