import Foundation
import Combine

// MARK: - Errors

enum FluxServiceError: LocalizedError {
    case invalidResponse
    case httpStatus(code: Int, bodyPreview: String?)
    case decodingFailed(underlying: Error)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "无效的服务器响应"
        case .httpStatus(let code, let preview):
            if let p = preview, !p.isEmpty { return "HTTP \(code): \(p)" }
            return "HTTP \(code)"
        case .decodingFailed(let e):
            return "JSON 解析失败: \(e.localizedDescription)"
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

    private var pollTask: Task<Void, Never>?
    private let session: URLSession

    init() {
        self.host = UserDefaults.standard.string(forKey: "flux_host") ?? "127.0.0.1"
        self.port = UserDefaults.standard.integer(forKey: "flux_port").nonZero ?? 8000

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 5
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        self.session = URLSession(configuration: config)
    }

    // MARK: - Polling (primary, always works)

    func startPolling() {
        stopPolling()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { break }
                await self.fetchState()
                try? await Task.sleep(for: .milliseconds(500))
            }
        }
        Task { await fetchStatus() }
    }

    func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
    }

    func reconnect() {
        stopPolling()
        startPolling()
        Task { await fetchStatus() }
    }

    func fetchState() async {
        do {
            let response: FluxResponse<FluxState> = try await get("api/v1/state")
            if let data = response.data {
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
            let response: FluxResponse<ServerStatus> = try await get("api/v1/status")
            self.serverStatus = response.data
        } catch {
            FluxLog.network.warn("fetchStatus 失败: \(error.localizedDescription)")
        }
    }

    func resetStamina() async {
        do {
            let _: FluxResponse<[String: String]> = try await post("api/v1/stamina/reset")
        } catch {
            FluxLog.network.warn("resetStamina 失败: \(error.localizedDescription)")
        }
    }

    func saveStamina() async {
        do {
            let _: FluxResponse<[String: String]> = try await post("api/v1/stamina/save")
        } catch {
            FluxLog.network.warn("saveStamina 失败: \(error.localizedDescription)")
        }
    }

    private func get<T: Decodable>(_ path: String) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        let (data, response) = try await session.data(from: url)
        try Self.validate(data: data, response: response)
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw FluxServiceError.decodingFailed(underlying: error)
        }
    }

    private func post<T: Decodable>(_ path: String) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        let (data, response) = try await session.data(for: request)
        try Self.validate(data: data, response: response)
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw FluxServiceError.decodingFailed(underlying: error)
        }
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
