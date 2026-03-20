import Foundation

/// 应用元数据 — 与视觉 token 分离，职责单一
enum FluxMeta {
    static let name    = "FluxChi"
    static let version = "1.2"
    static let schemaVersion = 1
    static let snapshotIntervalMs = 500
    static let githubURL = URL(string: "https://github.com/wujiajunhahah/Flux-hardware")!
}

// MARK: - Backward Compatibility

extension Flux {
    /// @available(*, deprecated, renamed: "FluxMeta")
    typealias App = FluxMeta
}
