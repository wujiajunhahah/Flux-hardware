import Foundation
import Combine

/// Manages REST API polling and SSE streaming to the FluxChi backend.
@MainActor
final class FluxService: ObservableObject {

    // MARK: - Published State

    @Published var state: FluxState?
    @Published var serverStatus: ServerStatus?
    @Published var isConnected = false
    @Published var connectionError: String?

    // MARK: - Configuration

    var baseURL: URL {
        URL(string: "http://\(host):\(port)")!
    }

    @Published var host: String {
        didSet { UserDefaults.standard.set(host, forKey: "flux_host") }
    }
    @Published var port: Int {
        didSet { UserDefaults.standard.set(port, forKey: "flux_port") }
    }

    private var sseTask: Task<Void, Never>?
    private var pollTimer: Timer?

    // MARK: - Init

    init() {
        self.host = UserDefaults.standard.string(forKey: "flux_host") ?? "localhost"
        self.port = UserDefaults.standard.integer(forKey: "flux_port").nonZero ?? 8000
    }

    // MARK: - SSE Stream

    func startSSE() {
        stopSSE()
        sseTask = Task { [weak self] in
            guard let self else { return }
            let url = self.baseURL.appendingPathComponent("api/v1/stream")
            var request = URLRequest(url: url)
            request.timeoutInterval = .infinity

            do {
                let (stream, response) = try await URLSession.shared.bytes(for: request)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    self.connectionError = "Server returned non-200"
                    self.isConnected = false
                    return
                }

                self.isConnected = true
                self.connectionError = nil

                var dataBuffer = ""
                for try await line in stream.lines {
                    if Task.isCancelled { break }

                    if line.hasPrefix("data: ") {
                        dataBuffer = String(line.dropFirst(6))
                    } else if line.isEmpty && !dataBuffer.isEmpty {
                        self.parseSSEData(dataBuffer)
                        dataBuffer = ""
                    }
                }
            } catch {
                if !Task.isCancelled {
                    self.connectionError = error.localizedDescription
                    self.isConnected = false
                    try? await Task.sleep(for: .seconds(3))
                    if !Task.isCancelled {
                        self.startSSE()
                    }
                }
            }
        }
    }

    func stopSSE() {
        sseTask?.cancel()
        sseTask = nil
    }

    private func parseSSEData(_ json: String) {
        guard let data = json.data(using: .utf8) else { return }
        do {
            let decoded = try JSONDecoder().decode(FluxState.self, from: data)
            self.state = decoded
        } catch {
            print("[SSE] Parse error: \(error)")
        }
    }

    // MARK: - REST Polling (fallback)

    func startPolling(interval: TimeInterval = 1.0) {
        stopPolling()
        pollTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                await self?.fetchState()
            }
        }
        Task { await fetchStatus() }
    }

    func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    func fetchState() async {
        do {
            let response: FluxResponse<FluxState> = try await get("api/v1/state")
            if let data = response.data {
                self.state = data
                self.isConnected = true
                self.connectionError = nil
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
            print("[API] Status error: \(error)")
        }
    }

    // MARK: - Control

    func resetStamina() async {
        let _: FluxResponse<[String: String]>? = try? await post("api/v1/stamina/reset")
    }

    func saveStamina() async {
        let _: FluxResponse<[String: String]>? = try? await post("api/v1/stamina/save")
    }

    // MARK: - Networking

    private func get<T: Decodable>(_ path: String) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func post<T: Decodable>(_ path: String) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(T.self, from: data)
    }
}

// MARK: - Helpers

private extension Int {
    var nonZero: Int? { self == 0 ? nil : self }
}
