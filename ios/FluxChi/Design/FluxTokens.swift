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
}
