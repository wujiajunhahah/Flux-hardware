import SwiftUI

/// 暗色渐变卡片样式 —— 与分享卡 PNG 对齐的品牌视觉。
///
/// 设计目标：作为 Dashboard 的"作品级"主卡片视觉，与 `.ultraThinMaterial` 系统风格形成对比。
/// 用在需要强识别 + 高信息密度的场景（今日 hero、关键 metric 卡）。
///
/// 用法：
/// ```swift
/// MyContent()
///     .modifier(FluxDarkCardStyle())
/// ```
struct FluxDarkCardStyle: ViewModifier {
    var cornerRadius: CGFloat = 28
    var padding: CGFloat = 22
    /// 主题色 —— 渐变末端混入的品牌色（默认 accent）
    var tint: Color = Flux.Colors.accent
    var tintIntensity: Double = 0.55

    func body(content: Content) -> some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                LinearGradient(
                    colors: [
                        Color.black,
                        tint.opacity(tintIntensity)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.06), lineWidth: 0.5)
            }
            .shadow(color: .black.opacity(0.18), radius: 14, x: 0, y: 6)
    }
}

extension View {
    /// 应用 FluxDarkCard 视觉样式（暗渐变 + 圆角 + 描边 + 阴影）。
    func fluxDarkCard(
        cornerRadius: CGFloat = 28,
        padding: CGFloat = 22,
        tint: Color = Flux.Colors.accent,
        tintIntensity: Double = 0.55
    ) -> some View {
        modifier(FluxDarkCardStyle(
            cornerRadius: cornerRadius,
            padding: padding,
            tint: tint,
            tintIntensity: tintIntensity
        ))
    }
}

// MARK: - 配套小元素

/// 暗卡上的小标签（白色 0.5/0.6 透明度 + tracking）。
struct FluxDarkLabel: View {
    let text: String
    var icon: String?

    var body: some View {
        HStack(spacing: 6) {
            if let icon {
                Image(systemName: icon)
                    .font(.system(size: 11, weight: .semibold))
            }
            Text(text)
                .font(.system(size: 11, weight: .semibold))
                .tracking(0.8)
                .textCase(.uppercase)
        }
        .foregroundStyle(.white.opacity(0.65))
    }
}

/// 暗卡上的指标块（小 label + 大 mono 值）。
struct FluxDarkMetric: View {
    let label: String
    let value: String
    var icon: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                if let icon {
                    Image(systemName: icon)
                        .font(.system(size: 9, weight: .semibold))
                }
                Text(label)
                    .font(.system(size: 10, weight: .medium))
                    .tracking(0.5)
                    .textCase(.uppercase)
            }
            .foregroundStyle(.white.opacity(0.55))

            Text(value)
                .font(.system(size: 18, weight: .semibold, design: .rounded))
                .foregroundStyle(.white)
                .contentTransition(.numericText())
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// 暗卡上的胶囊 chip（白 0.15 背景）。
struct FluxDarkChip: View {
    let text: String
    var icon: String?
    var tint: Color = .white

    var body: some View {
        HStack(spacing: 5) {
            if let icon {
                Image(systemName: icon)
                    .font(.system(size: 10, weight: .semibold))
            }
            Text(text)
                .font(.system(size: 11, weight: .semibold))
        }
        .foregroundStyle(tint.opacity(0.95))
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(
            tint == .white ? Color.white.opacity(0.15) : tint.opacity(0.18),
            in: Capsule()
        )
    }
}
