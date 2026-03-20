import Foundation
import Combine

// MARK: - Errors

enum FluxServiceError: LocalizedError {
    case invalidResponse
    case httpStatus(code: Int, bodyPreview: String?)
    case decodingFailed(underlying: Error)
    case envelopeFailed(code: String, message: String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "无效的服务器响应"
        case .httpStatus(let code, let preview):
            if let p = preview, !p.isEmpty { return "HTTP \(code): \(p)" }
            return "HTTP \(code)"
        case .decodingFailed(let e):
            return "JSON 解析失败: \(e.localizedDescription)"
        case .envelopeFailed(_, let message):
            return message.isEmpty ? "服务返回错误" : message
        }
    }
}

@MainActor
final class FluxService: ObservableObject {

    @Published var state: FluxState?
    @Published var serverStatus: ServerStatus?
    @Published var isConnected = false
    @Published var connectionError: String?

    /// 个性化管理器引用，由 App 层注入
    weak var personalization: PersonalizationManager?

    /// 经过个性化校准的 stamina 值
    var personalizedStaminaValue: Double {
        guard let raw = state?.stamina?.value else { return 0 }
        return personalization?.personalizedStamina(raw) ?? raw
    }

    var baseURL: URL {
        URL(string: "http://\(host):\(port)")!
    }

    @Published var host: String {
        didSet { UserDefaults.standard.set(host, forKey: "flux_host") }
    }
    @Published var port: Int {
        didSet { UserDefaults.standard.set(port, forKey: "flux_port") }
    }

    /// 短请求（REST）；长连接用 `streamSession`
    private let session: URLSession
    /// SSE：避免沿用 5s request timeout
    private let streamSession: URLSession
    private var wifiTransportTask: Task<Void, Never>?

    init() {
        self.host = UserDefaults.standard.string(forKey: "flux_host") ?? "127.0.0.1"
        self.port = UserDefaults.standard.integer(forKey: "flux_port").nonZero ?? 8000

        let short = URLSessionConfiguration.default
        short.timeoutIntervalForRequest = 5
        short.requestCachePolicy = .reloadIgnoringLocalCacheData
        self.session = URLSession(configuration: short)

        let long = URLSessionConfiguration.default
        long.timeoutIntervalForRequest = 0
        long.timeoutIntervalForResource = 60 * 60 * 24
        long.requestCachePolicy = .reloadIgnoringLocalCacheData
        self.streamSession = URLSession(configuration: long)
    }

    // MARK: - Wi‑Fi 传输：优先 SSE，失败后轮询

    func startPolling() {
        stopPolling()
        wifiTransportTask = Task { [weak self] in
            guard let self else { return }
            await self.runWiFiTransport()
        }
        Task { await fetchStatus() }
    }

    func stopPolling() {
        wifiTransportTask?.cancel()
        wifiTransportTask = nil
    }

    func reconnect() {
        stopPolling()
        startPolling()
        Task { await fetchStatus() }
    }

    /// 先连 `/api/v1/stream`（SSE）；任意失败或取消后改用轮询，直到任务取消。
    private func runWiFiTransport() async {
        await streamUntilSSEEnds()
        guard !Task.isCancelled else { return }
        FluxLog.network.info("SSE 不可用或已结束，回退到轮询")
        await runPollingLoop()
    }

    private func streamUntilSSEEnds() async {
        let url = baseURL.appendingPathComponent("api/v1/stream")
        var request = URLRequest(url: url)
        request.timeoutInterval = .infinity

        do {
            let (bytes, response) = try await streamSession.bytes(for: request)
            guard let http = response as? HTTPURLResponse else { return }
            guard (200...299).contains(http.statusCode) else { return }

            var pendingData: String?

            for try await line in bytes.lines {
                if Task.isCancelled { return }

                let line = line.trimmingCharacters(in: .whitespacesAndNewlines)

                if line.isEmpty {
                    if let json = pendingData {
                        pendingData = nil
                        applyStateFromSSEPayload(json)
                    }
                    continue
                }

                if line.hasPrefix(":") { continue }

                if line.hasPrefix("data:") {
                    let rest = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                    pendingData = String(rest)
                }
            }
        } catch is CancellationError {
            return
        } catch {
            FluxLog.network.warn("SSE: \(error.localizedDescription)")
        }
    }

    private func applyStateFromSSEPayload(_ json: String) {
        guard let data = json.data(using: .utf8) else { return }
        do {
            let decoded = try JSONDecoder().decode(FluxState.self, from: data)
            self.state = decoded
            if !self.isConnected {
                self.isConnected = true
                self.connectionError = nil
            }
        } catch {
            FluxLog.network.warn("SSE 解析 state 失败: \(error.localizedDescription)")
        }
    }

    private func runPollingLoop() async {
        while !Task.isCancelled {
            await fetchState()
            try? await Task.sleep(for: .milliseconds(500))
        }
    }

    func fetchState() async {
        do {
            let envelope: FluxResponse<FluxState> = try await requestEnvelope("api/v1/state")
            let data = try Self.unwrapEnvelope(envelope)
            if let data {
                self.state = data
                if !self.isConnected {
                    self.isConnected = true
                    self.connectionError = nil
                }
            }
        } catch {
            self.isConnected = false
            self.connectionError = error.localizedDescription
        }
    }

    func fetchStatus() async {
        do {
            let envelope: FluxResponse<ServerStatus> = try await requestEnvelope("api/v1/status")
            self.serverStatus = try Self.unwrapEnvelope(envelope)
        } catch {
            FluxLog.network.warn("fetchStatus 失败: \(error.localizedDescription)")
        }
    }

    func resetStamina() async {
        do {
            let envelope: FluxResponse<[String: String]> = try await requestEnvelope("api/v1/stamina/reset", method: "POST")
            _ = try Self.unwrapEnvelope(envelope)
        } catch {
            FluxLog.network.warn("resetStamina 失败: \(error.localizedDescription)")
        }
    }

    func saveStamina() async {
        do {
            let envelope: FluxResponse<[String: String]> = try await requestEnvelope("api/v1/stamina/save", method: "POST")
            _ = try Self.unwrapEnvelope(envelope)
        } catch {
            FluxLog.network.warn("saveStamina 失败: \(error.localizedDescription)")
        }
    }

    // MARK: - REST 信封

    private func requestEnvelope<T: Decodable>(
        _ path: String,
        method: String = "GET"
    ) async throws -> FluxResponse<T> {
        let url = baseURL.appendingPathComponent(path)
        var req = URLRequest(url: url)
        req.httpMethod = method
        let (data, response) = try await session.data(for: req)
        try Self.validate(data: data, response: response)
        do {
            return try JSONDecoder().decode(FluxResponse<T>.self, from: data)
        } catch {
            throw FluxServiceError.decodingFailed(underlying: error)
        }
    }

    /// `ok == false` 时抛出；`ok == true` 且 `data == nil` 时返回 nil（不抛错）。
    private static func unwrapEnvelope<T>(_ envelope: FluxResponse<T>) throws -> T? {
        if envelope.ok {
            return envelope.data
        }
        let code = envelope.error ?? "error"
        let message = envelope.message ?? envelope.error ?? "请求失败"
        throw FluxServiceError.envelopeFailed(code: code, message: message)
    }

    private static func validate(data: Data, response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else {
            throw FluxServiceError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            let preview = String(data: data, encoding: .utf8).map { String($0.prefix(160)) }
            throw FluxServiceError.httpStatus(code: http.statusCode, bodyPreview: preview)
        }
    }
}

private extension Int {
    var nonZero: Int? { self == 0 ? nil : self }
}
