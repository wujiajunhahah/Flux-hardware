import SwiftUI

/// 续航状态 → 颜色的**唯一实现**（主 App + Live Activity / Widget Extension 共用）。
/// 修改配色时只改此文件，并与 `StaminaState`（`FluxModels`）的 `rawValue` 保持一致。
enum StaminaStatePalette {

    /// 与 `StaminaState` 的 `String(rawValue)` 对齐；未知值回退为灰。
    static func color(forRawState raw: String) -> Color {
        switch raw.lowercased() {
        case "focused":    return .red
        case "fading":     return .orange
        case "depleted":   return .red.opacity(0.6)
        case "recovering": return .green
        default:           return .gray
        }
    }
}
