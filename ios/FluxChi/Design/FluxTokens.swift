import SwiftUI

enum Flux {

    // MARK: - Colors

    enum Colors {
        static let accent = Color.red
        static let success = Color.green
        static let warning = Color.orange

        static func forStaminaState(_ state: StaminaState) -> Color {
            switch state {
            case .focused:    return .red
            case .fading:     return .orange
            case .depleted:   return .red.opacity(0.6)
            case .recovering: return .green
            }
        }

        static func forUrgency(_ value: Double) -> Color {
            value >= 0.7 ? .red : value >= 0.5 ? .orange : .primary
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

    // MARK: - Text (文字层次)

    enum Text {
        /// 主要文字 - 标准可读性
        static let primary = Color.primary
        /// 次要文字 - 辅助信息
        static let secondary = Color.secondary
        /// 第三级文字 - 占位符等
        static let tertiary = Color(.tertiaryLabel)

        /// 暗化文字 - Widget 中使用
        static let dim = Color(white: 0.45)
        /// 静音文字 - 更暗
        static let muted = Color(white: 0.55)

        /// 强调色文字
        static let accent = Colors.accent
        /// 成功色文字
        static let success = Colors.success
        /// 警告色文字
        static let warning = Colors.warning
    }

    // MARK: - Backgrounds (背景层次)

    enum Backgrounds {
        /// 主背景 - 系统背景色
        static let primary = Color(uiColor: .systemBackground)
        /// 次要背景 - 分组背景
        static let secondary = Color(uiColor: .secondarySystemBackground)
        /// 第三级背景 - 用于更深层次
        static let tertiary = Color(uiColor: .tertiarySystemBackground)

        /// 抬升背景 - 浮层效果
        static let elevated = Color(white: 0.05, opacity: 1)
        /// 遮罩背景 - 半透明黑色
        static let overlay = Color.black.opacity(0.6)

        /// 休息模式专用背景 - 深绿色调
        static let rest = Color(red: 0.04, green: 0.12, blue: 0.08)
    }

    // MARK: - Sizes (尺寸系统)

    enum Sizes {
        // MARK: Icon Sizes

        static let iconSmall: CGFloat = 12
        static let iconMedium: CGFloat = 16
        static let iconLarge: CGFloat = 24
        static let iconXLarge: CGFloat = 32

        // MARK: Ring Sizes

        static let ringSmall: CGFloat = 72
        static let ringMedium: CGFloat = 88
        static let ringLarge: CGFloat = 120
        static let ringXLarge: CGFloat = 200

        // MARK: Text Sizes

        static let textLabel: CGFloat = 9
        static let textBody: CGFloat = 14
        static let textHeadline: CGFloat = 22
        static let textDisplay: CGFloat = 44

        // MARK: Stroke Widths

        static let strokeThin: CGFloat = 0.5
        static let strokeNormal: CGFloat = 1
        static let strokeThick: CGFloat = 2
        static let strokeBold: CGFloat = 4
    }

    // MARK: - Opacity (不透明度)

    enum Opacity {
        /// 极淡 - 几乎透明
        static let xLight: Double = 0.01
        /// 淡 - 轻微可见
        static let light: Double = 0.06
        /// 中等 - 标准半透明
        static let medium: Double = 0.12
        /// 半强 - 明显半透明
        static let semiStrong: Double = 0.18
        /// 强 - 高可见度
        static let strong: Double = 0.3
        /// 极强 - 接近不透明
        static let xStrong: Double = 0.5
    }

    // MARK: - Animation (动画)

    enum Animation {
        /// 快速动画 - 轻微反馈
        static let fastDuration: Double = 0.15
        /// 中速动画 - 标准过渡
        static let mediumDuration: Double = 0.6
        /// 慢速动画 - 大场景过渡
        static let slowDuration: Double = 0.8

        /// 缓出动画
        static let easeOut = SwiftUI.Animation.easeOut(duration: 0.15)

        /// 弹簧动画
        static let spring = SwiftUI.Animation.spring(response: 0.6, dampingFraction: 0.7)

        /// 创建缓出动画（自定义时长）
        static func easeOut(duration: Double) -> SwiftUI.Animation {
            .easeOut(duration: duration)
        }

        /// 创建弹簧动画（自定义参数）
        static func spring(response: Double, dampingFraction: Double) -> SwiftUI.Animation {
            .spring(response: response, dampingFraction: dampingFraction)
        }
    }

    // MARK: - Shadows (阴影)

    enum Shadows {
        /// 阴影配置 - 小阴影
        static func small(color: Color = .black.opacity(0.1)) -> [Color: CGFloat] {
            [color: 1]  // 简化表示，实际使用 .shadow()
        }

        /// 阴影配置 - 中阴影
        static func medium(color: Color = .black.opacity(0.15)) -> [Color: CGFloat] {
            [color: 1]
        }

        /// 阴影配置 - 大阴影
        static func large(color: Color = .black.opacity(0.2)) -> [Color: CGFloat] {
            [color: 1]
        }

        /// 发光颜色
        static func glowColor(base: Color = Colors.accent) -> Color {
            base.opacity(0.3)
        }
    }
}
