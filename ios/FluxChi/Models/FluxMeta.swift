import Foundation

/// 应用元数据 — 与视觉 token 分离，职责单一
enum FluxMeta {
    static let name    = "FluxChi"
    static let version = "1.2"
    static let schemaVersion = 1
    /// 快照采样间隔。1000ms 在 stamina 视觉曲线（约 0.05 Hz 实际变化频率）下视觉无差异，
    /// 但相比 500ms 减半内存压力（segment.snapshots 关系数组 + SwiftData dirty graph）。
    /// 5 处使用都从这个常量 derive（SessionManager / SummaryEngine / NLPStatsExtractor /
    /// ExportManager），改这一个数字全局生效。
    static let snapshotIntervalMs = 1000
    // swiftlint:disable:next force_unwrapping
    static let githubURL = URL(string: "https://github.com/wujiajunhahah/Flux-hardware")!
}

// MARK: - Backward Compatibility

extension Flux {
    /// @available(*, deprecated, renamed: "FluxMeta")
    typealias App = FluxMeta
}
