import Foundation
import OSLog
import SwiftUI

// MARK: - Log Level

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

// MARK: - Log Category

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

// MARK: - Log Entry

/// 日志条目模型 - 包含源码位置用于快速定位
struct FluxLogEntry: Identifiable, Codable, Hashable {
    let id: UUID
    let timestamp: Date
    let level: FluxLogLevel
    let category: FluxLogCategory
    let message: String
    let errorDescription: String?

    // 源码定位 - debug 核心能力
    let file: String
    let function: String
    let line: Int

    init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        level: FluxLogLevel,
        category: FluxLogCategory,
        message: String,
        errorDescription: String? = nil,
        file: String = "",
        function: String = "",
        line: Int = 0
    ) {
        self.id = id
        self.timestamp = timestamp
        self.level = level
        self.category = category
        self.message = message
        self.errorDescription = errorDescription
        self.file = file
        self.function = function
        self.line = line
    }

    /// 文件名（不含路径）
    var fileName: String {
        (file as NSString).lastPathComponent
    }

    /// 源码位置简写：FileName.swift:42
    var sourceLocation: String {
        guard !file.isEmpty else { return "" }
        return "\(fileName):\(line)"
    }

    // MARK: - Formatting

    private static let compactFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    private static let detailedFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return f
    }()

    /// 单行格式 - 控制台输出
    func formatCompact() -> String {
        let time = Self.compactFormatter.string(from: timestamp)
        let loc = sourceLocation.isEmpty ? "" : " \(sourceLocation)"
        let err = errorDescription.map { " | \($0)" } ?? ""
        return "[\(time)] [\(level.label)] [\(category.rawValue)]\(loc) \(message)\(err)"
    }

    /// 详细格式 - 导出用
    func formatDetailed() -> String {
        let time = Self.detailedFormatter.string(from: timestamp)
        var result = "[\(time)] \(level.label) - \(category.rawValue)"
        if !sourceLocation.isEmpty {
            result += " @ \(sourceLocation) \(function)"
        }
        result += "\n\(message)"
        if let error = errorDescription {
            result += "\nError: \(error)"
        }
        return result
    }
}

// MARK: - Log Config

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
        minimumLevel: .warn,
        enableConsole: false,
        maxMemoryEntries: 500,
        maxFileEntries: 5000,
        enableFilePersistence: true
    )
}

// MARK: - Log Export Options (唯一定义)

struct FluxLogExportOptions {
    let minimumLevel: FluxLogLevel?
    let categories: Set<FluxLogCategory>?
    let limit: Int?
    let keyword: String?

    static let all = FluxLogExportOptions(
        minimumLevel: nil, categories: nil, limit: nil, keyword: nil
    )

    static let errorsOnly = FluxLogExportOptions(
        minimumLevel: .error, categories: nil, limit: nil, keyword: nil
    )

    static let bleOnly = FluxLogExportOptions(
        minimumLevel: nil, categories: [.ble], limit: nil, keyword: nil
    )
}

// MARK: - Category Proxy

/// 便捷分类代理 - `FluxLog.ble.info("connected")`
struct FluxLogProxy {
    let category: FluxLogCategory

    func debug(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        FluxLog.debug(message, category: category, file: file, function: function, line: line)
    }

    func info(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        FluxLog.info(message, category: category, file: file, function: function, line: line)
    }

    func warn(_ message: String, file: String = #file, function: String = #function, line: Int = #line) {
        FluxLog.warn(message, category: category, file: file, function: function, line: line)
    }

    func error(_ message: String, error: Error? = nil, file: String = #file, function: String = #function, line: Int = #line) {
        FluxLog.error(message, category: category, error: error, file: file, function: function, line: line)
    }
}

// MARK: - FluxLog (统一入口 - 替代全局函数 + BLE 专用方法)

/// 全局日志入口 - 用法:
/// ```
/// FluxLog.info("启动完成", category: .app)
/// FluxLog.ble.debug("Frame received")
/// FluxLog.error("导出失败", category: .export, error: err)
/// ```
enum FluxLog {
    // 分类代理 - 一键引用
    static let ble         = FluxLogProxy(category: .ble)
    static let nlp         = FluxLogProxy(category: .nlp)
    static let ml          = FluxLogProxy(category: .ml)
    static let session     = FluxLogProxy(category: .session)
    static let performance = FluxLogProxy(category: .performance)
    static let export      = FluxLogProxy(category: .export)
    static let app         = FluxLogProxy(category: .app)
    static let network     = FluxLogProxy(category: .network)
    static let storage     = FluxLogProxy(category: .storage)
    static let ui          = FluxLogProxy(category: .ui)
    static let live        = FluxLogProxy(category: .liveActivity)

    static func debug(_ message: String, category: FluxLogCategory = .app,
                       file: String = #file, function: String = #function, line: Int = #line) {
        Task { @MainActor in
            FluxLogger.shared.log(level: .debug, message: message, category: category,
                                   file: file, function: function, line: line)
        }
    }

    static func info(_ message: String, category: FluxLogCategory = .app,
                      file: String = #file, function: String = #function, line: Int = #line) {
        Task { @MainActor in
            FluxLogger.shared.log(level: .info, message: message, category: category,
                                   file: file, function: function, line: line)
        }
    }

    static func warn(_ message: String, category: FluxLogCategory = .app,
                      file: String = #file, function: String = #function, line: Int = #line) {
        Task { @MainActor in
            FluxLogger.shared.log(level: .warn, message: message, category: category,
                                   file: file, function: function, line: line)
        }
    }

    static func error(_ message: String, category: FluxLogCategory = .app, error: Error? = nil,
                       file: String = #file, function: String = #function, line: Int = #line) {
        Task { @MainActor in
            FluxLogger.shared.log(level: .error, message: message, category: category, error: error,
                                   file: file, function: function, line: line)
        }
    }
}

// MARK: - FluxLogger (核心引擎)

@MainActor
final class FluxLogger: ObservableObject {

    // MARK: - Singleton

    static let shared = FluxLogger()

    // MARK: - Published

    @Published private(set) var entries: [FluxLogEntry] = []
    @Published private(set) var config: FluxLogConfig

    // MARK: - Private

    private let fileManager = FileManager.default
    private let osLog = OSLog(subsystem: "com.fluxchi.logger", category: "FluxLog")

    /// 持久化节流 - 防止高频写入
    private var pendingSave = false
    private let saveInterval: TimeInterval = 5.0 // 最多 5 秒写一次磁盘

    private var logFileURL: URL {
        let dir = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return dir.appendingPathComponent("fluxchi_logs.json")
    }

    // MARK: - Init

    private init(config: FluxLogConfig = .debug) {
        self.config = config
        loadFromFile()
    }

    // MARK: - Configuration

    func updateConfig(_ newConfig: FluxLogConfig) {
        self.config = newConfig
        log(level: .info, message: "日志配置已更新: minimumLevel=\(newConfig.minimumLevel.label)",
            category: .app, file: #file, function: #function, line: #line)
    }

    // MARK: - Core Log Method

    func log(
        level: FluxLogLevel,
        message: String,
        category: FluxLogCategory,
        error: Error? = nil,
        file: String = #file,
        function: String = #function,
        line: Int = #line
    ) {
        guard level >= config.minimumLevel else { return }

        let entry = FluxLogEntry(
            timestamp: Date(),
            level: level,
            category: category,
            message: message,
            errorDescription: error?.localizedDescription,
            file: file,
            function: function,
            line: line
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

        // 节流持久化 - 不再每条都写磁盘
        if config.enableFilePersistence {
            scheduleSave()
        }
    }

    // MARK: - Throttled Persistence

    private func scheduleSave() {
        guard !pendingSave else { return }
        pendingSave = true

        Task.detached(priority: .background) { [weak self] in
            try? await Task.sleep(for: .seconds(self?.saveInterval ?? 5.0))
            await self?.performSave()
        }
    }

    private func performSave() {
        pendingSave = false
        let entriesToSave = Array(entries.suffix(config.maxFileEntries))
        let url = logFileURL

        // 在后台线程写文件
        Task.detached(priority: .background) {
            do {
                let encoder = JSONEncoder()
                encoder.dateEncodingStrategy = .iso8601
                let data = try encoder.encode(entriesToSave)
                try data.write(to: url, options: .atomic)
            } catch {
                print("[FluxLogger] 保存日志失败: \(error)")
            }
        }
    }

    private func loadFromFile() {
        guard config.enableFilePersistence,
              fileManager.fileExists(atPath: logFileURL.path) else { return }

        do {
            let data = try Data(contentsOf: logFileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let loaded = try decoder.decode([FluxLogEntry].self, from: data)
            entries = Array(loaded.suffix(config.maxMemoryEntries))
        } catch {
            print("[FluxLogger] 加载日志失败: \(error)")
        }
    }

    // MARK: - Query

    func fetchEntries(
        limit: Int? = nil,
        category: FluxLogCategory? = nil,
        level: FluxLogLevel? = nil
    ) -> [FluxLogEntry] {
        var filtered = entries

        if let category { filtered = filtered.filter { $0.category == category } }
        if let level { filtered = filtered.filter { $0.level >= level } }
        if let limit { filtered = Array(filtered.suffix(limit)) }

        return filtered
    }

    func searchEntries(keyword: String) -> [FluxLogEntry] {
        guard !keyword.isEmpty else { return entries }
        return entries.filter {
            $0.message.localizedCaseInsensitiveContains(keyword) ||
            $0.errorDescription?.localizedCaseInsensitiveContains(keyword) == true ||
            $0.fileName.localizedCaseInsensitiveContains(keyword) ||
            $0.function.localizedCaseInsensitiveContains(keyword)
        }
    }

    // MARK: - Export

    func exportEntries(options: FluxLogExportOptions = .all) -> [FluxLogEntry] {
        var filtered = entries

        if let level = options.minimumLevel {
            filtered = filtered.filter { $0.level >= level }
        }
        if let categories = options.categories {
            filtered = filtered.filter { categories.contains($0.category) }
        }
        if let keyword = options.keyword, !keyword.isEmpty {
            filtered = filtered.filter {
                $0.message.localizedCaseInsensitiveContains(keyword) ||
                $0.errorDescription?.localizedCaseInsensitiveContains(keyword) == true
            }
        }
        if let limit = options.limit {
            filtered = Array(filtered.suffix(limit))
        }

        return filtered
    }

    func exportAsJSON(options: FluxLogExportOptions = .all) throws -> Data {
        let filtered = exportEntries(options: options)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(filtered)
    }

    func exportAsText(options: FluxLogExportOptions = .all) -> String {
        exportEntries(options: options)
            .map { $0.formatDetailed() }
            .joined(separator: "\n---\n")
    }

    /// 导出到临时文件并返回 URL
    func exportToFile(format: ExportFormat, options: FluxLogExportOptions = .all) throws -> URL {
        let dateStr = Self.exportDateFormatter.string(from: Date())

        switch format {
        case .json:
            let data = try exportAsJSON(options: options)
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("fluxchi_logs_\(dateStr).json")
            try data.write(to: url)
            return url

        case .text:
            let text = exportAsText(options: options)
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("fluxchi_logs_\(dateStr).txt")
            try text.write(to: url, atomically: true, encoding: .utf8)
            return url
        }
    }

    enum ExportFormat {
        case json, text
    }

    private static let exportDateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd_HHmmss"
        return f
    }()

    // MARK: - Management

    func clearLogs() {
        entries.removeAll()
        try? fileManager.removeItem(at: logFileURL)
    }

    func trimOldLogs(keepingRecent count: Int) {
        if entries.count > count {
            entries = Array(entries.suffix(count))
        }
    }

    /// 强制立即保存（退出前调用）
    func flushToDisk() {
        pendingSave = false
        let entriesToSave = Array(entries.suffix(config.maxFileEntries))
        let url = logFileURL
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(entriesToSave)
            try data.write(to: url, options: .atomic)
        } catch {
            print("[FluxLogger] flush 失败: \(error)")
        }
    }
}

// MARK: - Backward Compatibility (迁移期保留)

/// 兼容旧调用方 - 逐步替换为 FluxLog.*
extension FluxLogger {
    // 保留旧的直接调用方式
    func debug(_ message: String, category: FluxLogCategory = .app,
               file: String = #file, function: String = #function, line: Int = #line) {
        log(level: .debug, message: message, category: category, file: file, function: function, line: line)
    }

    func info(_ message: String, category: FluxLogCategory = .app,
              file: String = #file, function: String = #function, line: Int = #line) {
        log(level: .info, message: message, category: category, file: file, function: function, line: line)
    }

    func warn(_ message: String, category: FluxLogCategory = .app,
              file: String = #file, function: String = #function, line: Int = #line) {
        log(level: .warn, message: message, category: category, file: file, function: function, line: line)
    }

    func error(_ message: String, category: FluxLogCategory = .app, error: Error? = nil,
               file: String = #file, function: String = #function, line: Int = #line) {
        log(level: .error, message: message, category: category, error: error, file: file, function: function, line: line)
    }
}
