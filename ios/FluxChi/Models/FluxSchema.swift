import Foundation
import SwiftData

/// SwiftData 版本化 schema。
///
/// **当前状态**：仅有 `SchemaV1`，无迁移阶段。下一次给 `Session/Segment/FluxSnapshot/UserFeedback`
/// 加字段时，新建 `SchemaV2` 并把 `Session.self` 等替换为 V2 的同名类型，再向 `FluxMigrationPlan.stages`
/// 追加迁移段（多数情况下 `MigrationStage.lightweight`）。
///
/// **为什么必须有这个文件**：
/// `Scene.modelContainer(for:)` 不接受 `MigrationPlan`，只能用 `ModelContainer(for:migrationPlan:)`
/// 手动构建。一旦给 `@Model` 加字段而没有 plan，SwiftData 会直接抛 migration 错并阻止启动。
enum FluxSchemaV1: VersionedSchema {
    static var versionIdentifier: Schema.Version { Schema.Version(1, 0, 0) }

    static var models: [any PersistentModel.Type] {
        [Session.self, Segment.self, FluxSnapshot.self, UserFeedback.self]
    }
}

enum FluxMigrationPlan: SchemaMigrationPlan {
    static var schemas: [any VersionedSchema.Type] {
        [FluxSchemaV1.self]
    }

    /// 迁移阶段。新增 schema 版本时往这里追加 `MigrationStage.lightweight(...)`
    /// 或 `.custom(...)`（前者适合纯添加字段，后者用于字段重命名/拆分）。
    static var stages: [MigrationStage] { [] }
}

// MARK: - Container Builder

enum FluxModelContainer {
    /// 构建带迁移计划的 `ModelContainer`。失败时降级为内存容器（避免启动崩溃但提示用户）。
    @MainActor
    static func makeShared() -> ModelContainer {
        let schema = Schema(FluxSchemaV1.models)
        let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)
        do {
            return try ModelContainer(
                for: schema,
                migrationPlan: FluxMigrationPlan.self,
                configurations: config
            )
        } catch {
            FluxLog.storage.error("SwiftData 容器初始化失败，降级为内存模式（数据不会保存）", error: error)
            // 退化路径：让 app 起得来，至少 UI 能渲染；用户在设置里看到"未连接 + 历史为空"
            let fallback = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
            // 内存容器若依然失败，只能 fatalError —— 至少有 OSLog 留痕
            // swiftlint:disable:next force_try
            return try! ModelContainer(for: schema, configurations: fallback)
        }
    }
}
