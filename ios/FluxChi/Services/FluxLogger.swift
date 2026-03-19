import Foundation
import OSLog
import SwiftUI

// MARK: - Flux Logger System

/// 日志级别 - 支持比较和排序
enum FluxLogLevel: Int, Comparable, CaseIterable, Codable {
    case debug = 0
    case info  = 1
    case warn  = 2
    case error = 3

    var icon: String {
        switch self {
        case .debug: return "ladybug.fill"
        case .info:  return "info.circle.fill"
        case .warn:  return "exclamationmark.triangle.fill"
        case .error: return "xmark.circle.fill"
        }
    }

    var color: Color {
        switch self {
        case .debug: return .blue
        case .info:  return .cyan
        case .warn:  return .orange
        case .error: return .red
        }
    }

    var label: String {
        switch self {
        case .debug: return "DEBUG"
        case .info:  return "INFO"
        case .warn:  return "WARN"
        case .error: return "ERROR"
        }
    }

    static func < (lhs: FluxLogLevel, rhs: FluxLogLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// 日志分类 - 按功能模块划分
enum FluxLogCategory: String, CaseIterable, Codable {
    case ble            = "BLE"
    case nlp            = "NLP"
    case liveActivity   = "LiveActivity"
    case ml             = "ML"
    case session        = "Session"
    case performance    = "Performance"
    case export         = "Export"
    case app            = "App"
    case network        = "Network"
    case storage        = "Storage"
    case ui             = "UI"

    var icon: String {
        switch self {
        case .ble:          return "antenna.radiowaves.left.and.right"
        case .nlp:          return "text.bubble"
        case .liveActivity: return "app.dashed"
        case .ml:           return "brain.head.profile"
        case .session:      return "clock.arrow.circlepath"
        case .performance:  return "gauge.with.dots.needle.67percent"
        case .export:       return "square.and.arrow.up"
        case .app:          return "app.badge"
        case .network:      return "network"
        case .storage:      return "externaldrive"
        case .ui:           return "rectangle.stack"
        }
    }

    var color: Color {
        switch self {
        case .ble:          return .blue
        case .nlp:          return .purple
        case .liveActivity: return .orange
        case .ml:           return .pink
        case .session:      return .green
        case .performance:  return .yellow
        case .export:       return .indigo
        case .app:          return .gray
        case .network:      return .cyan
        case .storage:      return .brown
        case .ui:           return .mint
        }
    }
}

/// 日志条目模型
struct FluxLogEntry: Identifiable, Codable, Hashable {
    let id: UUID
    let timestamp: Date
    let level: FluxLogLevel
    let category: FluxLogCategory
    let message: String
    let errorDescription: String?

    init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        level: FluxLogLevel,
        category: FluxLogCategory,
        message: String,
        errorDescription: String? = nil
    ) {
        self.id = id
        self.timestamp = timestamp
        self.level = level
        self.category = category
        self.message = message
        self.errorDescription = errorDescription
    }

    /// 格式化为单行文本
    func formatCompact() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        let time = formatter.string(from: timestamp)
        let errorSuffix = errorDescription.map { " \($0)" } ?? ""
        return "[\(time)] [\(level.label)] [\(category.rawValue)] \(message)\(errorSuffix)"
    }

    /// 格式化为详细文本
    func formatDetailed() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        let time = formatter.string(from: timestamp)
        var result = """
        [\(time)] \(level.label) - \(category.rawValue)
        \(message)
        """
        if let error = errorDescription {
            result += "\nError: \(error)"
        }
        return result
    }
}

/// 日志配置
struct FluxLogConfig {
    var minimumLevel: FluxLogLevel
    var enableConsole: Bool
    var maxMemoryEntries: Int
    var maxFileEntries: Int
    var enableFilePersistence: Bool

    static let debug = FluxLogConfig(
        minimumLevel: .debug,
        enableConsole: true,
        maxMemoryEntries: 2000,
        maxFileEntries: 50000,
        enableFilePersistence: true
    )

    static let production = FluxLogConfig(
        minimumLevel: .error,
        enableConsole: false,
        maxMemoryEntries: 500,
        maxFileEntries: 5000,
        enableFilePersistence: true
    )
}

// MARK: - Category Proxy (便捷访问)

/// 类别代理 - 提供"一键引用"的便捷日志接口
struct FluxLoggerCategoryProxy {
    let category: FluxLogCategory

    func debug(_ message: String) {
        Task { @MainActor in
            FluxLogger.shared.debug(message, category: category)
        }
    }

    func info(_ message: String) {
        Task { @MainActor in
            FluxLogger.shared.info(message, category: category)
        }
    }

    func warn(_ message: String) {
        Task { @MainActor in
            FluxLogger.shared.warn(message, category: category)
        }
    }

    func error(_ message: String, error: Error? = nil) {
        Task { @MainActor in
            FluxLogger.shared.error(message, category: category, error: error)
        }
    }
}

// MARK: - Main Logger Class

@MainActor
final class FluxLogger: ObservableObject {

    // MARK: - Singleton

    static let shared = FluxLogger()

    // MARK: - Published State

    @Published private(set) var entries: [FluxLogEntry] = []
    @Published private(set) var config: FluxLogConfig

    // MARK: - Private

    private let fileManager = FileManager.default
    private var logFileURL: URL {
        let documentsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsDir.appendingPathComponent("fluxchi_logs.json")
    }

    private let osLog = OSLog(subsystem: "com.fluxchi.logger", category: "FluxLog")

    // MARK: - Category Proxies (便捷访问 - "一键引用")

    static let ble = FluxLoggerCategoryProxy(category: .ble)
    static let nlp = FluxLoggerCategoryProxy(category: .nlp)
    static let performance = FluxLoggerCategoryProxy(category: .performance)
    static let ml = FluxLoggerCategoryProxy(category: .ml)
    static let session = FluxLoggerCategoryProxy(category: .session)
    static let liveActivity = FluxLoggerCategoryProxy(category: .liveActivity)
    static let export = FluxLoggerCategoryProxy(category: .export)
    static let app = FluxLoggerCategoryProxy(category: .app)
    static let network = FluxLoggerCategoryProxy(category: .network)
    static let storage = FluxLoggerCategoryProxy(category: .storage)
    static let ui = FluxLoggerCategoryProxy(category: .ui)

    // MARK: - Init

    private init(config: FluxLogConfig = .debug) {
        self.config = config
        loadFromFile()
    }

    // MARK: - Configuration

    func updateConfig(_ newConfig: FluxLogConfig) {
        self.config = newConfig
        info("日志配置已更新: minimumLevel=\(newConfig.minimumLevel.label)", category: .app)
    }

    // MARK: - Logging Methods

    private func log(
        level: FluxLogLevel,
        message: String,
        category: FluxLogCategory,
        error: Error? = nil
    ) {
        guard level >= config.minimumLevel else { return }

        let entry = FluxLogEntry(
            timestamp: Date(),
            level: level,
            category: category,
            message: message,
            errorDescription: error?.localizedDescription
        )

        // 内存存储
        entries.append(entry)
        if entries.count > config.maxMemoryEntries {
            entries.removeFirst(entries.count - config.maxMemoryEntries)
        }

        // 控制台输出
        if config.enableConsole {
            print(entry.formatCompact())
        }

        // OSLog
        let osLogType: OSLogType = {
            switch level {
            case .debug: return .debug
            case .info:  return .info
            case .warn:  return .default
            case .error: return .error
            }
        }()
        os_log("%{public}@", log: osLog, type: osLogType, entry.formatCompact())

        // 异步持久化
        if config.enableFilePersistence {
            Task.detached(priority: .background) { [weak self] in
                await self?.saveToFile()
            }
        }
    }

    func debug(_ message: String, category: FluxLogCategory = .app) {
        log(level: .debug, message: message, category: category)
    }

    func info(_ message: String, category: FluxLogCategory = .app) {
        log(level: .info, message: message, category: category)
    }

    func warn(_ message: String, category: FluxLogCategory = .app) {
        log(level: .warn, message: message, category: category)
    }

    func error(_ message: String, category: FluxLogCategory = .app, error: Error? = nil) {
        log(level: .error, message: message, category: category, error: error)
    }

    // MARK: - BLE Specialized Methods

    /// 记录 BLE 扫描事件
    static func logBLEScan(_ message: String, level: FluxLogLevel = .info) {
        Task { @MainActor in
            shared.log(level: level, message: message, category: .ble)
        }
    }

    /// 记录 BLE 连接事件
    static func logBLEConnect(_ message: String, level: FluxLogLevel = .info) {
        Task { @MainActor in
            shared.log(level: level, message: message, category: .ble)
        }
    }

    /// 记录 BLE 断开事件
    static func logBLEDisconnect(_ message: String, error: Error? = nil, level: FluxLogLevel = .warn) {
        Task { @MainActor in
            shared.log(level: level, message: message, category: .ble, error: error)
        }
    }

    /// 记录 BLE 数据事件（采样使用）
    static func logBLEData(_ message: String, level: FluxLogLevel = .debug) {
        Task { @MainActor in
            shared.log(level: level, message: message, category: .ble)
        }
    }

    // MARK: - Query Methods

    /// 获取日志条目
    func fetchEntries(
        limit: Int? = nil,
        category: FluxLogCategory? = nil,
        level: FluxLogLevel? = nil
    ) -> [FluxLogEntry] {
        var filtered = entries

        if let category = category {
            filtered = filtered.filter { $0.category == category }
        }

        if let level = level {
            filtered = filtered.filter { $0.level >= level }
        }

        if let limit = limit {
            filtered = Array(filtered.suffix(limit))
        }

        return filtered
    }

    /// 搜索日志
    func searchEntries(keyword: String) -> [FluxLogEntry] {
        entries.filter {
            $0.message.localizedCaseInsensitiveContains(keyword) ||
            $0.errorDescription?.localizedCaseInsensitiveContains(keyword) == true
        }
    }

    /// 获取指定时间范围的日志
    func fetchEntries(from startDate: Date, to endDate: Date) -> [FluxLogEntry] {
        entries.filter { $0.timestamp >= startDate && $0.timestamp <= endDate }
    }

    // MARK: - Export Methods

    /// 导出日志为 JSON
    func exportLogs() throws -> URL {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(entries)
        let dateStr = DateFormatter().string(from: Date()).replacingOccurrences(of: " ", with: "_").replacingOccurrences(of: ":", with: "")
        let filename = "fluxchi_logs_\(dateStr).json"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)

        try data.write(to: url)
        return url
    }

    /// 导出日志为文本
    func exportLogsAsText(
        minimumLevel: FluxLogLevel? = nil,
        categories: Set<FluxLogCategory>? = nil,
        limit: Int? = nil
    ) throws -> URL {
        let filtered = fetchEntries(limit: limit, category: categories?.first, level: minimumLevel)

        let text = filtered.map { $0.formatDetailed() }.joined(separator: "\n---\n")

        let dateStr = DateFormatter().string(from: Date()).replacingOccurrences(of: " ", with: "_").replacingOccurrences(of: ":", with: "")
        let filename = "fluxchi_logs_\(dateStr).txt"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)

        try text.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    // MARK: - Management

    /// 清空所有日志
    func clearLogs() {
        entries.removeAll()
        info("日志已清空", category: .app)
        try? fileManager.removeItem(at: logFileURL)
    }

    /// 清空过期日志（保留最近的 N 条）
    func trimOldLogs(keepingRecent count: Int) {
        if entries.count > count {
            entries = Array(entries.suffix(count))
        }
    }

    // MARK: - File Persistence

    private nonisolated func saveToFile() async {
        guard await config.enableFilePersistence else { return }

        // 获取当前需要保存的条目
        let entriesToSave = await entries.suffix(await config.maxFileEntries)
        let logURL = await logFileURL

        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(Array(entriesToSave))
            try data.write(to: logURL, options: .atomic)
        } catch {
            print("[FluxLogger] 保存日志失败: \(error)")
        }
    }

    private func loadFromFile() {
        guard config.enableFilePersistence else { return }

        guard fileManager.fileExists(atPath: logFileURL.path) else { return }

        do {
            let data = try Data(contentsOf: logFileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601

            let loaded = try decoder.decode([FluxLogEntry].self, from: data)

            // 只加载最近的 maxMemoryEntries 条
            entries = Array(loaded.suffix(config.maxMemoryEntries))

            info("已从文件加载 \(entries.count) 条日志", category: .app)
        } catch {
            print("[FluxLogger] 加载日志失败: \(error)")
        }
    }
}

// MARK: - Global Convenience Functions

/// 全局便捷函数
func LogDebug(_ message: String, category: FluxLogCategory = .app) {
    Task { @MainActor in
        FluxLogger.shared.debug(message, category: category)
    }
}

func LogInfo(_ message: String, category: FluxLogCategory = .app) {
    Task { @MainActor in
        FluxLogger.shared.info(message, category: category)
    }
}

func LogWarn(_ message: String, category: FluxLogCategory = .app) {
    Task { @MainActor in
        FluxLogger.shared.warn(message, category: category)
    }
}

func LogError(_ message: String, category: FluxLogCategory = .app, error: Error? = nil) {
    Task { @MainActor in
        FluxLogger.shared.error(message, category: category, error: error)
    }
}

// MARK: - Export Options

/// 日志导出选项
struct LogExportOptions {
    let minimumLevel: FluxLogLevel?
    let categories: Set<FluxLogCategory>?
    let limit: Int?
    let keyword: String?

    static let `default` = LogExportOptions(
        minimumLevel: nil,
        categories: nil,
        limit: nil,
        keyword: nil
    )

    static let errorsOnly = LogExportOptions(
        minimumLevel: .error,
        categories: nil,
        limit: nil,
        keyword: nil
    )

    static let bleOnly = LogExportOptions(
        minimumLevel: nil,
        categories: [.ble],
        limit: nil,
        keyword: nil
    )
}
