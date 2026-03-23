import SwiftUI
import UIKit

// MARK: - Flux Design System

/// FluxChi 统一设计规范
/// 所有视觉常量集中在此，禁止在视图中硬编码数值。
enum Flux {

    // MARK: - Colors (语义色板)

    enum Colors {
        // 品牌主色
        static let accent = Color.red

        // 语义色
        static let success = Color.green
        static let warning = Color.orange
        static let info = Color.blue
        static let destructive = Color(UIColor.systemRed)  // 与 accent 区分：系统红偏冷

        // 边界 & 分割线
        static let border = Color.primary.opacity(0.1)
        static let divider = Color.primary.opacity(0.08)

        /// 根据 StaminaState 枚举返回对应颜色（实现见 `StaminaStatePalette`，与 Extension 共用）
        static func forStaminaState(_ state: StaminaState) -> Color {
            StaminaStatePalette.color(forRawState: state.rawValue)
        }

        /// String 版 — 兼容 rawValue 传入场景，逐步迁移后删除
        @available(*, deprecated, message: "Use forStaminaState(_ state: StaminaState) instead")
        static func forStaminaState(_ state: String) -> Color {
            if let parsed = StaminaState(rawValue: state.lowercased()) {
                return StaminaStatePalette.color(forRawState: parsed.rawValue)
            }
            return StaminaStatePalette.color(forRawState: state)
        }

        /// 根据续航数值返回颜色
        static func forStaminaValue(_ value: Double) -> Color {
            if value > 60 { return forStaminaState(.focused) }
            if value > 30 { return forStaminaState(.fading) }
            return forStaminaState(.depleted)
        }

        static func forUrgency(_ value: Double) -> Color {
            value >= 0.7 ? .red : value >= 0.5 ? .orange : .primary
        }
    }

    // MARK: - Label (文字层次 - 避免与 SwiftUI.Text 冲突)

    enum Label {
        /// 主要文字
        static let primary = Color.primary
        /// 次要文字
        static let secondary = Color.secondary
        /// 第三级文字
        static let tertiary = Color(UIColor.tertiaryLabel)
        /// 暗化文字 - Widget 中使用
        static let dim = Color(white: 0.45)
        /// 静音文字
        static let muted = Color(white: 0.55)
        /// 强调色文字
        static let accent = Colors.accent
        /// 成功色文字
        static let success = Colors.success
        /// 警告色文字
        static let warning = Colors.warning
    }

    // MARK: - Surface (背景层次 - 语义更清晰)

    enum Surface {
        /// 主背景
        static let primary = Color(UIColor.systemBackground)
        /// 分组背景
        static let secondary = Color(UIColor.secondarySystemBackground)
        /// 更深层次背景
        static let tertiary = Color(UIColor.tertiarySystemBackground)
        /// 抬升背景 - sheet/popover 等浮层（dark 模式下比 secondary 更亮）
        static let elevated = Color(UIColor.secondarySystemGroupedBackground)
        /// 遮罩背景
        static let overlay = Color.black.opacity(0.6)
        /// 休息模式背景 - 深绿色调
        static let rest = Color(red: 0.04, green: 0.12, blue: 0.08)
        /// 沉浸专注全屏（与 `ActiveSessionView` 一致）
        static let focusImmersive = Color.black
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
        static let xl:     CGFloat = 24
    }

    // MARK: - Sizes

    enum Sizes {
        // Icon
        static let iconSmall:  CGFloat = 12
        static let iconMedium: CGFloat = 16
        static let iconLarge:  CGFloat = 24
        static let iconXLarge: CGFloat = 32

        // Ring
        static let ringSmall:  CGFloat = 72
        static let ringMedium: CGFloat = 88
        static let ringLarge:  CGFloat = 120
        static let ringXLarge: CGFloat = 200

        // Text
        static let textLabel:    CGFloat = 9
        static let textBody:     CGFloat = 14
        static let textHeadline: CGFloat = 22
        static let textDisplay:  CGFloat = 44

        // Stroke
        static let strokeThin:   CGFloat = 0.5
        static let strokeNormal: CGFloat = 1
        static let strokeThick:  CGFloat = 2
        static let strokeBold:   CGFloat = 4
    }

    // MARK: - Opacity

    enum Opacity {
        static let xLight:     Double = 0.01
        static let light:      Double = 0.06
        static let medium:     Double = 0.12
        static let semiStrong: Double = 0.18
        static let strong:     Double = 0.3
        static let xStrong:    Double = 0.5
    }

    // MARK: - Motion (避免与 SwiftUI.Animation 冲突)

    enum Motion {
        static let fast: SwiftUI.Animation = .easeOut(duration: 0.15)
        static let standard: SwiftUI.Animation = .easeOut(duration: 0.3)
        static let spring: SwiftUI.Animation = .spring(response: 0.6, dampingFraction: 0.7)
        static let slow: SwiftUI.Animation = .easeInOut(duration: 0.8)

        static func easeOut(duration: Double) -> SwiftUI.Animation {
            .easeOut(duration: duration)
        }

        static func spring(response: Double, dampingFraction: Double) -> SwiftUI.Animation {
            .spring(response: response, dampingFraction: dampingFraction)
        }
    }

    // MARK: - Shadow (实际可用的阴影修饰符)

    enum Shadow {
        /// 小阴影 - 按钮、小卡片
        static func small(_ view: some View, color: Color = .black.opacity(0.08)) -> some View {
            view.shadow(color: color, radius: 2, x: 0, y: 1)
        }

        /// 中阴影 - 卡片、浮层
        static func medium(_ view: some View, color: Color = .black.opacity(0.12)) -> some View {
            view.shadow(color: color, radius: 8, x: 0, y: 4)
        }

        /// 大阴影 - 模态、弹窗
        static func large(_ view: some View, color: Color = .black.opacity(0.16)) -> some View {
            view.shadow(color: color, radius: 16, x: 0, y: 8)
        }

        /// 发光效果
        static func glow(_ view: some View, color: Color = Colors.accent, radius: CGFloat = 10) -> some View {
            view.shadow(color: color.opacity(0.3), radius: radius, x: 0, y: 2)
        }
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

// MARK: - Device

extension Flux {
    enum Device {
        /// iPad 上浮动 Tab Bar 与 `NavigationStack` 大标题易叠加顶部留白；用于 Tab 根页标题模式等分支。
        static var isPad: Bool {
            UIDevice.current.userInterfaceIdiom == .pad
        }
    }
}

// MARK: - Backward Compatibility (逐步迁移后删除)

extension Flux {
    /// @available(*, deprecated, renamed: "Label")
    typealias Text = Label
    /// @available(*, deprecated, renamed: "Surface")
    typealias Backgrounds = Surface
}
