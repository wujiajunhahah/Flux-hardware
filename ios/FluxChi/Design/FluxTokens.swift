import SwiftUI

enum Flux {

    // MARK: - Colors (BIOSORA-inspired warm palette)

    enum Colors {
        /// 主强调色 — 暖珊瑚（替代原纯红，更有机、温暖）
        static let accent = Color(red: 0.85, green: 0.50, blue: 0.38) // #D98061

        /// 语义色
        static let success = Color(red: 0.50, green: 0.70, blue: 0.52) // 鼠尾草绿
        static let warning = Color(red: 0.85, green: 0.68, blue: 0.42) // 暖琥珀

        /// 续航状态色 — 暖色系四阶
        static func forStaminaState(_ state: StaminaState) -> Color {
            switch state {
            case .focused:    return accent                                        // 珊瑚 — 专注
            case .fading:     return warning                                       // 琥珀 — 衰退
            case .depleted:   return Color(red: 0.62, green: 0.50, blue: 0.64)     // 薰衣草 — 耗尽
            case .recovering: return success                                       // 鼠尾草 — 恢复
            }
        }

        /// 根据续航数值返回颜色（HistoryView SessionRow 等）
        static func forStaminaValue(_ value: Double) -> Color {
            if value > 60 { return forStaminaState(.focused) }
            if value > 30 { return forStaminaState(.fading) }
            return forStaminaState(.depleted)
        }

        static func forUrgency(_ value: Double) -> Color {
            value >= 0.7 ? accent : value >= 0.5 ? warning : .primary
        }
    }

    // MARK: - Materials (Liquid Glass 风格统一材质)

    enum Materials {
        static let card: Material = .ultraThinMaterial
        static let sheet: Material = .regularMaterial
        static let overlay: Material = .thinMaterial

        /// iOS 26+ glass 材质，低版本 fallback ultraThinMaterial
        @ViewBuilder
        static func glass<S: Shape>(in shape: S) -> some View {
            if #available(iOS 26.0, *) {
                Rectangle().fill(.ultraThinMaterial).clipShape(shape)
            } else {
                Rectangle().fill(.ultraThinMaterial).clipShape(shape)
            }
        }
    }

    // MARK: - Shapes

    enum Shapes {
        static func capsule() -> Capsule { Capsule() }
        static func card() -> RoundedRectangle { RoundedRectangle(cornerRadius: Radius.large, style: .continuous) }
        static func smallCard() -> RoundedRectangle { RoundedRectangle(cornerRadius: Radius.medium, style: .continuous) }

        /// 同心圆形状（用于 Ring 等）
        static func concentric(outer: CGFloat, inner: CGFloat) -> some Shape {
            Circle().stroke(lineWidth: (outer - inner) / 2)
        }
    }

    // MARK: - Typography

    enum Typography {
        static func metric(_ size: CGFloat = 34) -> Font {
            .system(size: size, weight: .bold, design: .rounded)
        }

        static let mono      = Font.system(size: 13, design: .monospaced)
        static let monoSmall = Font.system(size: 10, weight: .semibold, design: .monospaced)
        static let section   = Font.caption.weight(.medium)
    }

    // MARK: - Spacing

    enum Spacing {
        static let section: CGFloat = 24
        static let group:   CGFloat = 16
        static let item:    CGFloat = 12
        static let inner:   CGFloat = 8
        static let tight:   CGFloat = 4
    }

    // MARK: - Radius

    enum Radius {
        static let small:  CGFloat = 8
        static let medium: CGFloat = 12
        static let large:  CGFloat = 16
    }

    // MARK: - App Constants

    enum App {
        static let name    = "FluxChi"
        static let version = "1.0"
        static let schemaVersion = 1
        static let snapshotIntervalMs = 500
        static let githubURL = URL(string: "https://github.com/wujiajunhahah/Flux-hardware")!
    }

    // MARK: - Formatters

    static func formatMinutes(_ m: Double) -> String {
        m < 1 ? "< 1" : "\(Int(m))"
    }

    static func formatDuration(_ seconds: TimeInterval) -> String {
        let h = Int(seconds) / 3600
        let m = (Int(seconds) % 3600) / 60
        let s = Int(seconds) % 60
        if h > 0 { return "\(h)h \(m)m" }
        if m > 0 { return "\(m)m \(s)s" }
        return "\(s)s"
    }

    static func formatDurationLong(_ seconds: TimeInterval) -> String {
        let h = Int(seconds) / 3600
        let m = (Int(seconds) % 3600) / 60
        if h > 0 { return "\(h) 小时 \(m) 分钟" }
        return "\(m) 分钟"
    }
}
