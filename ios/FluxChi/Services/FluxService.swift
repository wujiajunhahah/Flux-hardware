import Foundation
import Combine
import UIKit

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
    @Published private(set) var platformDeviceID: String?

    /// 个性化管理器引用，由 App 层注入
    weak var personalization: PersonalizationManager?

    /// 优先使用服务端 **融合** 续航（与 Web 主环一致），否则 EMG `stamina`；无有效值时为 `nil`。
    var personalizedDisplayStamina: Double? {
        guard let state else { return nil }
        if let fusion = state.fusion,
           fusion.source != "none",
           let fused = fusion.stamina {
            return personalization?.personalizedStamina(fused) ?? fused
        }
        if let raw = state.stamina?.value {
            return personalization?.personalizedStamina(raw) ?? raw
        }
        return nil
    }

    /// 向后兼容：无有效综合分时回退 0（UI 层应优先用 `personalizedDisplayStamina` + 占位符）
    var personalizedStaminaValue: Double {
        personalizedDisplayStamina ?? 0
    }

    /// 与 `personalizedDisplayStamina` 配套的展示用状态（融合优先）
    var displayStaminaState: StaminaState {
        guard let state else { return .focused }
        if let fusion = state.fusion,
           fusion.source != "none",
           fusion.stamina != nil,
           let s = StaminaState(rawValue: fusion.state) {
            return s
        }
        if let emg = state.stamina, let s = StaminaState(rawValue: emg.state) {
            return s
        }
        return .focused
    }

    var baseURL: URL {
        URL(string: "http://\(host):\(port)") ?? URL(string: "http://127.0.0.1:8000")!
    }

    var currentPlatformDeviceID: String? {
        platformSessionState?.deviceID
    }

    @Published var host: String {
        didSet { UserDefaults.standard.set(host, forKey: Self.hostDefaultsKey) }
    }
    @Published var port: Int {
        didSet { UserDefaults.standard.set(port, forKey: Self.portDefaultsKey) }
    }

    /// 短请求（REST）；长连接用 `streamSession`
    private let session: URLSession
    /// Blob 上传使用更宽松的超时，避免长 session JSON 在弱网下被 5s request timeout 误杀。
    private let uploadSession: URLSession
    /// SSE：避免沿用 5s request timeout
    private let streamSession: URLSession
    private var wifiTransportTask: Task<Void, Never>?
    private var platformSessionState: PlatformAuthSession? {
        didSet {
            persistPlatformSession()
            platformDeviceID = platformSessionState?.deviceID
        }
    }

    private static let hostDefaultsKey = "flux_host"
    private static let portDefaultsKey = "flux_port"
    private static let platformClientDeviceKeyDefaultsKey = "flux_platform_client_device_key"
    private static let platformSessionDefaultsKey = "flux_platform_session"

    init() {
        self.host = UserDefaults.standard.string(forKey: Self.hostDefaultsKey) ?? "127.0.0.1"
        self.port = UserDefaults.standard.integer(forKey: Self.portDefaultsKey).nonZero ?? 8000

        let short = URLSessionConfiguration.default
        short.timeoutIntervalForRequest = 5
        short.requestCachePolicy = .reloadIgnoringLocalCacheData
        self.session = URLSession(configuration: short)

        let upload = URLSessionConfiguration.default
        upload.timeoutIntervalForRequest = 60
        upload.timeoutIntervalForResource = 300
        upload.requestCachePolicy = .reloadIgnoringLocalCacheData
        self.uploadSession = URLSession(configuration: upload)

        let long = URLSessionConfiguration.default
        long.timeoutIntervalForRequest = 0
        long.timeoutIntervalForResource = 60 * 60 * 24
        long.requestCachePolicy = .reloadIgnoringLocalCacheData
        self.streamSession = URLSession(configuration: long)

        self.platformSessionState = Self.loadPlatformSession()
        self.platformDeviceID = self.platformSessionState?.deviceID
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
        let url = buildURL(path: "api/v1/stream")
        var request = URLRequest(url: url)
        request.timeoutInterval = .infinity

        do {
            let (bytes, response) = try await streamSession.bytes(for: request)
            guard let http = response as? HTTPURLResponse else { return }
            guard (200...299).contains(http.statusCode) else { return }

            // SSE：同一事件可含多行 `data:`，须用 `\n` 拼接（HTML 标准）；空行表示事件结束。
            var pendingDataLines: [String] = []

            for try await line in bytes.lines {
                if Task.isCancelled { return }

                let line = line.trimmingCharacters(in: .whitespacesAndNewlines)

                if line.isEmpty {
                    if !pendingDataLines.isEmpty {
                        let json = pendingDataLines.joined(separator: "\n")
                        pendingDataLines.removeAll(keepingCapacity: true)
                        applyStateFromSSEPayload(json)
                    }
                    continue
                }

                if line.hasPrefix(":") { continue }

                if line.hasPrefix("data:") {
                    var rest = String(line.dropFirst(5))
                    if rest.first == " " { rest.removeFirst() }
                    pendingDataLines.append(rest)
                }
            }
            if !pendingDataLines.isEmpty {
                applyStateFromSSEPayload(pendingDataLines.joined(separator: "\n"))
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
            let decoded = try Self.makeDecoder().decode(FluxState.self, from: data)
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

    // MARK: - Platform API

    func fetchPlatformBootstrap(channel: String = "stable") async throws -> PlatformBootstrapData {
        let session = try await ensurePlatformSession()
        let envelope: FluxResponse<PlatformBootstrapData> = try await requestPlatformEnvelope(
            "v1/sync/bootstrap",
            queryItems: [
                URLQueryItem(name: "device_id", value: session.deviceID),
                URLQueryItem(name: "platform", value: "ios"),
                URLQueryItem(name: "channel", value: channel),
            ]
        )
        return try Self.requireData(envelope)
    }

    func resolvePlatformDeviceID() async throws -> String {
        try await ensurePlatformSession().deviceID
    }

    func updatePlatformProfile(
        baseVersion: Int,
        calibrationOffset: Double,
        estimatedAccuracy: Double,
        trainingCount: Int,
        activeModelReleaseID: String?,
        summary: PlatformProfileSummary
    ) async throws -> PlatformProfileState {
        let body = try Self.makeEncoder().encode(
            PlatformUpdateProfileRequest(
                baseVersion: baseVersion,
                calibrationOffset: calibrationOffset,
                estimatedAccuracy: estimatedAccuracy,
                trainingCount: trainingCount,
                activeModelReleaseID: activeModelReleaseID,
                summary: summary
            )
        )
        let envelope: FluxResponse<PlatformUpdateProfileResponse> = try await requestPlatformEnvelope(
            "v1/profile",
            method: "PUT",
            body: body
        )
        return try Self.requireData(envelope).profileState
    }

    func updatePlatformDeviceCalibration(
        deviceID: String,
        baseVersion: Int,
        deviceName: String,
        sensorProfile: PlatformSensorProfile,
        calibrationOffset: Double
    ) async throws -> PlatformDeviceCalibrationState {
        let body = try Self.makeEncoder().encode(
            PlatformUpdateDeviceCalibrationRequest(
                baseVersion: baseVersion,
                deviceName: deviceName,
                sensorProfile: sensorProfile,
                calibrationOffset: calibrationOffset
            )
        )
        let envelope: FluxResponse<PlatformUpdateDeviceCalibrationResponse> = try await requestPlatformEnvelope(
            "v1/devices/\(deviceID)/calibration",
            method: "PUT",
            body: body
        )
        return try Self.requireData(envelope).deviceCalibration
    }

    func createPlatformSession(
        _ payload: PlatformCreateSessionRequest,
        idempotencyKey: String
    ) async throws -> PlatformCreateSessionResponse {
        let body = try Self.makeEncoder().encode(payload)
        let envelope: FluxResponse<PlatformCreateSessionResponse> = try await requestPlatformEnvelope(
            "v1/sessions",
            method: "POST",
            body: body,
            headers: ["Idempotency-Key": idempotencyKey]
        )
        return try Self.requireData(envelope)
    }

    func uploadPlatformSessionBlob(
        uploadURL: String,
        payload: Data,
        contentType: String,
        method: String = "PUT"
    ) async throws {
        guard let url = URL(string: uploadURL) else {
            throw FluxServiceError.envelopeFailed(code: "invalid_upload_url", message: "上传地址无效")
        }
        guard method.uppercased() == "PUT" else {
            throw FluxServiceError.envelopeFailed(code: "unsupported_upload_method", message: "平台返回了不支持的上传方法")
        }
        var request = URLRequest(url: url)
        request.httpMethod = method.uppercased()
        request.httpBody = payload
        request.setValue(contentType, forHTTPHeaderField: "Content-Type")

        let (_, response) = try await uploadSession.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw FluxServiceError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            throw FluxServiceError.httpStatus(code: http.statusCode, bodyPreview: nil)
        }
    }

    func finalizePlatformSession(
        sessionID: String,
        objectKey: String,
        sizeBytes: Int,
        sha256: String
    ) async throws -> PlatformFinalizeSessionResponse {
        let body = try Self.makeEncoder().encode(
            PlatformFinalizeSessionRequest(
                status: "ready",
                blob: PlatformSessionBlobPayload(
                    objectKey: objectKey,
                    sizeBytes: sizeBytes,
                    sha256: sha256
                )
            )
        )
        let envelope: FluxResponse<PlatformFinalizeSessionResponse> = try await requestPlatformEnvelope(
            "v1/sessions/\(sessionID)",
            method: "PATCH",
            body: body
        )
        return try Self.requireData(envelope)
    }

    // MARK: - REST Helpers

    private func requestEnvelope<T: Decodable>(
        _ path: String,
        method: String = "GET",
        queryItems: [URLQueryItem] = [],
        body: Data? = nil,
        headers: [String: String] = [:]
    ) async throws -> FluxResponse<T> {
        let url = buildURL(path: path, queryItems: queryItems)
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.httpBody = body
        if body != nil {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        for (field, value) in headers {
            req.setValue(value, forHTTPHeaderField: field)
        }

        let (data, response) = try await session.data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw FluxServiceError.invalidResponse
        }

        let decodedEnvelope: FluxResponse<T>?
        let decodeError: Error?
        do {
            decodedEnvelope = try Self.makeDecoder().decode(FluxResponse<T>.self, from: data)
            decodeError = nil
        } catch {
            decodedEnvelope = nil
            decodeError = error
        }

        guard (200...299).contains(http.statusCode) else {
            if let envelope = decodedEnvelope {
                throw FluxServiceError.envelopeFailed(
                    code: envelope.resolvedErrorCode,
                    message: envelope.resolvedErrorMessage
                )
            }
            let preview = String(data: data, encoding: .utf8).map { String($0.prefix(160)) }
            throw FluxServiceError.httpStatus(code: http.statusCode, bodyPreview: preview)
        }

        guard let envelope = decodedEnvelope else {
            throw FluxServiceError.decodingFailed(underlying: decodeError ?? FluxServiceError.invalidResponse)
        }

        if !envelope.ok {
            throw FluxServiceError.envelopeFailed(
                code: envelope.resolvedErrorCode,
                message: envelope.resolvedErrorMessage
            )
        }

        return envelope
    }

    private func requestPlatformEnvelope<T: Decodable>(
        _ path: String,
        method: String = "GET",
        queryItems: [URLQueryItem] = [],
        body: Data? = nil,
        headers: [String: String] = [:],
        retryOnUnauthorized: Bool = true
    ) async throws -> FluxResponse<T> {
        let session = try await ensurePlatformSession()
        do {
            var authorizedHeaders = headers
            authorizedHeaders["Authorization"] = "Bearer \(session.accessToken)"
            return try await requestEnvelope(
                path,
                method: method,
                queryItems: queryItems,
                body: body,
                headers: authorizedHeaders
            )
        } catch {
            guard retryOnUnauthorized, Self.isUnauthorized(error) else {
                throw error
            }
            _ = try await ensurePlatformSession(forceRefresh: true)
            return try await requestPlatformEnvelope(
                path,
                method: method,
                queryItems: queryItems,
                body: body,
                headers: headers,
                retryOnUnauthorized: false
            )
        }
    }

    private func ensurePlatformSession(forceRefresh: Bool = false) async throws -> PlatformAuthSession {
        if !forceRefresh,
           let session = platformSessionState,
           session.accessTokenExpiresAt.timeIntervalSinceNow > 60 {
            return session
        }

        if let existing = platformSessionState, !existing.refreshToken.isEmpty {
            do {
                let refreshed = try await refreshPlatformSession(existing)
                platformSessionState = refreshed
                return refreshed
            } catch {
                FluxLog.network.warn("平台 token 刷新失败: \(error.localizedDescription)")
                if forceRefresh || Self.isUnauthorized(error) {
                    platformSessionState = nil
                }
            }
        }

        let authenticated = try await signInOrSignUpPlatform()
        platformSessionState = authenticated
        return authenticated
    }

    private func signInOrSignUpPlatform() async throws -> PlatformAuthSession {
        do {
            return try await authenticatePlatform(createUser: false)
        } catch {
            guard Self.errorCode(from: error) == "user_not_found" else {
                throw error
            }
        }

        do {
            return try await authenticatePlatform(createUser: true)
        } catch {
            if Self.errorCode(from: error) == "identity_already_exists" {
                return try await authenticatePlatform(createUser: false)
            }
            throw error
        }
    }

    private func authenticatePlatform(createUser: Bool) async throws -> PlatformAuthSession {
        let payload = makePlatformAuthRequest()
        let body = try Self.makeEncoder().encode(payload)
        let path = createUser ? "v1/auth/sign-up" : "v1/auth/sign-in"
        let envelope: FluxResponse<PlatformAuthData> = try await requestEnvelope(
            path,
            method: "POST",
            body: body
        )
        let data = try Self.requireData(envelope)
        let refreshToken = data.refreshToken ?? platformSessionState?.refreshToken ?? ""
        guard !refreshToken.isEmpty else {
            throw FluxServiceError.envelopeFailed(code: "bad_envelope", message: "缺少 refresh token")
        }
        return PlatformAuthSession(
            userID: data.userID,
            deviceID: data.deviceID,
            accessToken: data.accessToken,
            refreshToken: refreshToken,
            accessTokenExpiresAt: Date().addingTimeInterval(TimeInterval(data.expiresInSec))
        )
    }

    private func refreshPlatformSession(_ existing: PlatformAuthSession) async throws -> PlatformAuthSession {
        let body = try Self.makeEncoder().encode(
            PlatformRefreshRequest(refreshToken: existing.refreshToken)
        )
        let envelope: FluxResponse<PlatformAuthData> = try await requestEnvelope(
            "v1/auth/refresh",
            method: "POST",
            body: body
        )
        let data = try Self.requireData(envelope)
        return PlatformAuthSession(
            userID: existing.userID,
            deviceID: existing.deviceID,
            accessToken: data.accessToken,
            refreshToken: existing.refreshToken,
            accessTokenExpiresAt: Date().addingTimeInterval(TimeInterval(data.expiresInSec))
        )
    }

    private func makePlatformAuthRequest() -> PlatformAuthRequest {
        PlatformAuthRequest(
            provider: "apple",
            providerToken: platformProviderToken(),
            device: PlatformAuthDevice(
                clientDeviceKey: platformClientDeviceKey(),
                platform: "ios",
                deviceName: UIDevice.current.name,
                appVersion: Flux.App.version,
                osVersion: UIDevice.current.systemVersion
            )
        )
    }

    private func platformClientDeviceKey() -> String {
        let defaults = UserDefaults.standard
        if let key = defaults.string(forKey: Self.platformClientDeviceKeyDefaultsKey), !key.isEmpty {
            return key
        }
        let generated = UIDevice.current.identifierForVendor?.uuidString.lowercased()
            ?? UUID().uuidString.lowercased()
        defaults.set(generated, forKey: Self.platformClientDeviceKeyDefaultsKey)
        return generated
    }

    private func platformProviderToken() -> String {
        let deviceKey = platformClientDeviceKey()
        let bundleID = Bundle.main.bundleIdentifier ?? "fluxchi.ios"
        return "dev:\(bundleID).\(deviceKey)"
    }

    private func persistPlatformSession() {
        let defaults = UserDefaults.standard
        guard let platformSessionState else {
            defaults.removeObject(forKey: Self.platformSessionDefaultsKey)
            return
        }
        guard let data = try? Self.makeEncoder().encode(platformSessionState) else {
            return
        }
        defaults.set(data, forKey: Self.platformSessionDefaultsKey)
    }

    private func buildURL(path: String, queryItems: [URLQueryItem] = []) -> URL {
        var url = baseURL
        for component in path.split(separator: "/") {
            url.appendPathComponent(String(component))
        }
        guard !queryItems.isEmpty,
              var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            return url
        }
        components.queryItems = queryItems
        return components.url ?? url
    }

    private static func loadPlatformSession() -> PlatformAuthSession? {
        guard let data = UserDefaults.standard.data(forKey: platformSessionDefaultsKey) else {
            return nil
        }
        return try? makeDecoder().decode(PlatformAuthSession.self, from: data)
    }

    private static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    private static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    private static func unwrapEnvelope<T>(_ envelope: FluxResponse<T>) throws -> T? {
        if envelope.ok {
            return envelope.data
        }
        throw FluxServiceError.envelopeFailed(
            code: envelope.resolvedErrorCode,
            message: envelope.resolvedErrorMessage
        )
    }

    private static func requireData<T>(_ envelope: FluxResponse<T>) throws -> T {
        guard let data = try unwrapEnvelope(envelope) else {
            throw FluxServiceError.envelopeFailed(code: "bad_envelope", message: "服务返回缺少 data")
        }
        return data
    }

    private static func errorCode(from error: Error) -> String? {
        guard case let FluxServiceError.envelopeFailed(code, _) = error else {
            return nil
        }
        return code
    }

    private static func isUnauthorized(_ error: Error) -> Bool {
        if case let FluxServiceError.httpStatus(code, _) = error, code == 401 {
            return true
        }
        return errorCode(from: error) == "unauthorized"
    }
}

private extension Int {
    var nonZero: Int? { self == 0 ? nil : self }
}
