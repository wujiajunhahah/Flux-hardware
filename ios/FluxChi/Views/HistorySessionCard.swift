import SwiftUI

/// Oura/Whoop 风格的会话卡片：续航大圆环 + 时间范围 + sparkline + 状态徽章。
/// 用 `.ultraThinMaterial` 玻璃质感 + 系统圆角，原生 iOS 卡片视感。
struct HistorySessionCard: View {
    let session: Session

    private var timeRange: String {
        let fmt = DateFormatter()
        fmt.dateFormat = "HH:mm"
        let start = fmt.string(from: session.startedAt)
        let end = fmt.string(from: session.startedAt.addingTimeInterval(session.duration))
        return "\(start) – \(end)"
    }

    private var staminaColor: Color {
        guard let avg = session.avgStamina else { return Color.secondary }
        return Flux.Colors.forStaminaValue(avg)
    }

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            staminaRing
            content
            Spacer(minLength: 0)
            indicators
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .background {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(.ultraThinMaterial)
        }
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.06), lineWidth: 0.5)
        }
    }

    // MARK: - Stamina Ring

    private var staminaRing: some View {
        ZStack {
            Circle()
                .stroke(staminaColor.opacity(0.12), lineWidth: 4)
                .frame(width: 52, height: 52)

            if let avg = session.avgStamina {
                Circle()
                    .trim(from: 0, to: CGFloat(min(max(avg / 100, 0), 1)))
                    .stroke(
                        staminaColor,
                        style: StrokeStyle(lineWidth: 4, lineCap: .round)
                    )
                    .rotationEffect(.degrees(-90))
                    .frame(width: 52, height: 52)

                VStack(spacing: -2) {
                    Text("\(Int(avg))")
                        .font(.system(size: 16, weight: .semibold, design: .rounded))
                        .foregroundStyle(staminaColor)
                    Text("续航")
                        .font(.system(size: 8, weight: .medium))
                        .foregroundStyle(.tertiary)
                        .textCase(.uppercase)
                        .tracking(0.6)
                }
            } else {
                Image(systemName: "ellipsis")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - Content

    private var content: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(timeRange)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.primary)

            HStack(spacing: 6) {
                Image(systemName: "clock")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.tertiary)
                Text(Flux.formatDuration(session.duration))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)

                if let count = session.segmentCount, count > 0 {
                    Text("·")
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
                    Text("\(count) 段")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }

            sparkline
                .frame(height: 16)
                .padding(.top, 2)
        }
    }

    // MARK: - Sparkline

    @ViewBuilder
    private var sparkline: some View {
        let values = session.staminaCurve
        if values.count >= 2 {
            GeometryReader { geo in
                let maxV = values.max() ?? 100
                let minV = values.min() ?? 0
                let range = max(maxV - minV, 1)

                ZStack {
                    Path { path in
                        for (i, v) in values.enumerated() {
                            let x = geo.size.width * CGFloat(i) / CGFloat(max(values.count - 1, 1))
                            let y = geo.size.height * (1 - CGFloat((v - minV) / range))
                            if i == 0 { path.move(to: CGPoint(x: x, y: y)) }
                            else { path.addLine(to: CGPoint(x: x, y: y)) }
                        }
                    }
                    .stroke(staminaColor.opacity(0.7), lineWidth: 1.5)
                }
            }
        } else {
            Color.clear
        }
    }

    // MARK: - Right-side indicators

    private var indicators: some View {
        VStack(alignment: .trailing, spacing: 6) {
            if session.feedback == nil {
                feedbackPendingBadge
            }
            Image(systemName: session.source.icon)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.quaternary)
        }
    }

    private var feedbackPendingBadge: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(Flux.Colors.warning)
                .frame(width: 6, height: 6)
            Text("待反馈")
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(Flux.Colors.warning)
        }
    }
}
