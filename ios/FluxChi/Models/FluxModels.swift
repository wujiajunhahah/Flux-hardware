import Foundation

// MARK: - API Response Envelope

struct FluxResponse<T: Decodable>: Decodable {
    let ok: Bool
    let ts: TimeInterval?
    let requestID: String?
    let data: T?
    let message: String?
    let errorCode: String?
    let errorMessage: String?

    private enum CodingKeys: String, CodingKey {
        case ok
        case ts
        case requestID = "request_id"
        case data
        case error
        case message
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        ok = try container.decode(Bool.self, forKey: .ok)
        ts = try container.decodeIfPresent(TimeInterval.self, forKey: .ts)
        requestID = try container.decodeIfPresent(String.self, forKey: .requestID)
        data = try? container.decode(T.self, forKey: .data)
        message = try container.decodeIfPresent(String.self, forKey: .message)

        if let detail = try? container.decodeIfPresent(FluxErrorDetail.self, forKey: .error) {
            errorCode = detail.code
            errorMessage = detail.message
        } else if let code = try? container.decodeIfPresent(String.self, forKey: .error) {
            errorCode = code
            errorMessage = nil
        } else {
            errorCode = nil
            errorMessage = nil
        }
    }

    var resolvedErrorCode: String {
        errorCode ?? "error"
    }

    var resolvedErrorMessage: String {
        if let message, !message.isEmpty { return message }
        if let errorMessage, !errorMessage.isEmpty { return errorMessage }
        if let errorCode, !errorCode.isEmpty { return errorCode }
        return "请求失败"
    }
}

private struct FluxErrorDetail: Decodable {
    let code: String
    let message: String
}

// MARK: - Platform API Models

struct PlatformAuthRequest: Encodable {
    let provider: String
    let providerToken: String
    let device: PlatformAuthDevice

    enum CodingKeys: String, CodingKey {
        case provider
        case providerToken = "provider_token"
        case device
    }
}

struct PlatformAuthDevice: Encodable {
    let clientDeviceKey: String
    let platform: String
    let deviceName: String
    let appVersion: String
    let osVersion: String

    enum CodingKeys: String, CodingKey {
        case clientDeviceKey = "client_device_key"
        case platform
        case deviceName = "device_name"
        case appVersion = "app_version"
        case osVersion = "os_version"
    }
}

struct PlatformRefreshRequest: Encodable {
    let refreshToken: String

    enum CodingKeys: String, CodingKey {
        case refreshToken = "refresh_token"
    }
}

struct PlatformAuthData: Decodable {
    let userID: String
    let deviceID: String
    let accessToken: String
    let refreshToken: String?
    let expiresInSec: Int

    enum CodingKeys: String, CodingKey {
        case userID = "user_id"
        case deviceID = "device_id"
        case accessToken = "access_token"
        case refreshToken = "refresh_token"
        case expiresInSec = "expires_in_sec"
    }
}

struct PlatformAuthSession: Codable {
    let userID: String
    let deviceID: String
    let accessToken: String
    let refreshToken: String
    let accessTokenExpiresAt: Date
}

struct PlatformBootstrapData: Decodable {
    let serverTime: Date
    let profileState: PlatformProfileState
    let deviceCalibrations: [PlatformDeviceCalibrationState]
    let activeModelManifest: PlatformModelManifest?

    enum CodingKeys: String, CodingKey {
        case serverTime = "server_time"
        case profileState = "profile_state"
        case deviceCalibrations = "device_calibrations"
        case activeModelManifest = "active_model_manifest"
    }
}

struct PlatformProfileState: Decodable {
    let profileID: String
    let version: Int
    let calibrationOffset: Double
    let estimatedAccuracy: Double
    let trainingCount: Int
    let activeModelReleaseID: String?
    let summary: PlatformProfileSummary
    let updatedAt: Date

    enum CodingKeys: String, CodingKey {
        case profileID = "profile_id"
        case version
        case calibrationOffset = "calibration_offset"
        case estimatedAccuracy = "estimated_accuracy"
        case trainingCount = "training_count"
        case activeModelReleaseID = "active_model_release_id"
        case summary
        case updatedAt = "updated_at"
    }
}

struct PlatformProfileSummary: Codable {
    let retainedFeedbackCount: Int
    let avgAbsoluteError: Double?
    let lastFeedbackAt: Date?

    enum CodingKeys: String, CodingKey {
        case retainedFeedbackCount = "retained_feedback_count"
        case avgAbsoluteError = "avg_absolute_error"
        case lastFeedbackAt = "last_feedback_at"
    }
}

struct PlatformDeviceCalibrationState: Decodable {
    let deviceID: String
    let version: Int
    let deviceName: String
    let calibrationOffset: Double
    let updatedAt: Date

    enum CodingKeys: String, CodingKey {
        case deviceID = "device_id"
        case version
        case deviceName = "device_name"
        case calibrationOffset = "calibration_offset"
        case updatedAt = "updated_at"
    }
}

struct PlatformModelManifest: Decodable {
    let modelReleaseID: String
    let version: String
    let artifactURL: String?
    let publishedAt: Date?

    enum CodingKeys: String, CodingKey {
        case modelReleaseID = "model_release_id"
        case version
        case artifactURL = "artifact_url"
        case publishedAt = "published_at"
    }
}

struct PlatformUpdateProfileRequest: Encodable {
    let baseVersion: Int
    let calibrationOffset: Double
    let estimatedAccuracy: Double
    let trainingCount: Int
    let activeModelReleaseID: String?
    let summary: PlatformProfileSummary

    enum CodingKeys: String, CodingKey {
        case baseVersion = "base_version"
        case calibrationOffset = "calibration_offset"
        case estimatedAccuracy = "estimated_accuracy"
        case trainingCount = "training_count"
        case activeModelReleaseID = "active_model_release_id"
        case summary
    }
}

struct PlatformUpdateProfileResponse: Decodable {
    let profileState: PlatformProfileState

    enum CodingKeys: String, CodingKey {
        case profileState = "profile_state"
    }
}

struct PlatformSensorProfile: Codable {
    let channels: Int
    let sampleRateHz: Int
    let quality: Int?
    let relaxMean: [Double]?
    let mvcPeak: [Double]?
    let calibratedAt: Date?

    enum CodingKeys: String, CodingKey {
        case channels
        case sampleRateHz = "sample_rate_hz"
        case quality
        case relaxMean = "relax_mean"
        case mvcPeak = "mvc_peak"
        case calibratedAt = "calibrated_at"
    }
}

struct PlatformUpdateDeviceCalibrationRequest: Encodable {
    let baseVersion: Int
    let deviceName: String
    let sensorProfile: PlatformSensorProfile
    let calibrationOffset: Double

    enum CodingKeys: String, CodingKey {
        case baseVersion = "base_version"
        case deviceName = "device_name"
        case sensorProfile = "sensor_profile"
        case calibrationOffset = "calibration_offset"
    }
}

struct PlatformUpdateDeviceCalibrationResponse: Decodable {
    let deviceCalibration: PlatformDeviceCalibrationState

    enum CodingKeys: String, CodingKey {
        case deviceCalibration = "device_calibration"
    }
}

struct PlatformCreateSessionRequest: Encodable {
    let sessionID: String
    let deviceID: String
    let source: String
    let title: String?
    let startedAt: Date
    let endedAt: Date
    let durationSec: Int
    let snapshotCount: Int
    let schemaVersion: Int
    let contentType: String
    let sizeBytes: Int
    let sha256: String

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case deviceID = "device_id"
        case source
        case title
        case startedAt = "started_at"
        case endedAt = "ended_at"
        case durationSec = "duration_sec"
        case snapshotCount = "snapshot_count"
        case schemaVersion = "schema_version"
        case contentType = "content_type"
        case sizeBytes = "size_bytes"
        case sha256
    }
}

struct PlatformCreateSessionResponse: Decodable {
    let session: PlatformSessionState
    let upload: PlatformSessionUploadDescriptor?
}

struct PlatformSessionState: Decodable {
    let sessionID: String
    let status: String
    let downloadURL: String?

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case status
        case downloadURL = "download_url"
    }
}

struct PlatformSessionUploadDescriptor: Decodable {
    let objectKey: String
    let uploadURL: String
    let uploadMethod: String
    let contentType: String

    enum CodingKeys: String, CodingKey {
        case objectKey = "object_key"
        case uploadURL = "upload_url"
        case uploadMethod = "upload_method"
        case contentType = "content_type"
    }
}

struct PlatformFinalizeSessionRequest: Encodable {
    let status: String
    let blob: PlatformSessionBlobPayload
}

struct PlatformSessionBlobPayload: Encodable {
    let objectKey: String
    let sizeBytes: Int
    let sha256: String

    enum CodingKeys: String, CodingKey {
        case objectKey = "object_key"
        case sizeBytes = "size_bytes"
        case sha256
    }
}

struct PlatformFinalizeSessionResponse: Decodable {
    let session: PlatformSessionState
}

struct PlatformCreateFeedbackEventRequest: Encodable {
    let feedbackEventID: String
    let deviceID: String
    let sessionID: String?
    let predictedStamina: Int
    let actualStamina: Int
    let label: String
    let kss: Int?
    let note: String?
    let createdAt: Date

    enum CodingKeys: String, CodingKey {
        case feedbackEventID = "feedback_event_id"
        case deviceID = "device_id"
        case sessionID = "session_id"
        case predictedStamina = "predicted_stamina"
        case actualStamina = "actual_stamina"
        case label
        case kss
        case note
        case createdAt = "created_at"
    }
}

struct PlatformCreateFeedbackEventResponse: Decodable {
    let feedbackEvent: PlatformFeedbackEventState

    enum CodingKeys: String, CodingKey {
        case feedbackEvent = "feedback_event"
    }
}

struct PlatformFeedbackEventState: Decodable {
    let feedbackEventID: String
    let sessionID: String?
    let createdAt: Date?

    enum CodingKeys: String, CodingKey {
        case feedbackEventID = "feedback_event_id"
        case sessionID = "session_id"
        case createdAt = "created_at"
    }
}

// MARK: - State Update (WebSocket / SSE payload)

struct FluxState: Decodable, Equatable {
    let timestamp: TimeInterval
    let activity: String
    let confidence: Double
    let probabilities: [String: Double]
    let rms: [Double]
    let emgSampleCount: Int?
    let stamina: StaminaData?
    let decision: DecisionData?
    /// 与 Web 仪表盘一致：多源融合结果（无手环仅摄像头时尤为关键）
    let fusion: FusionPayload?
    /// 视觉快照 + `stale`（后端注入）
    let vision: VisionPayload?

    enum CodingKeys: String, CodingKey {
        case timestamp, activity, confidence, probabilities, rms
        case emgSampleCount = "emg_sample_count"
        case stamina, decision, fusion, vision
    }

    static func == (lhs: FluxState, rhs: FluxState) -> Bool {
        lhs.timestamp == rhs.timestamp
    }
}

// MARK: - Fusion (matches FusedReading.to_dict snake_case)

struct FusionPayload: Decodable, Equatable {
    let stamina: Double?
    let state: String
    let source: String
    let alerts: [String]?
    let emgWeight: Double?
    let visionWeight: Double?
    let visionFatigue: Double?
    let visionQuality: Double?

    enum CodingKeys: String, CodingKey {
        case stamina, state, source, alerts
        case emgWeight = "emg_weight"
        case visionWeight = "vision_weight"
        case visionFatigue = "vision_fatigue"
        case visionQuality = "vision_quality"
    }
}

// MARK: - Vision snapshot (+ stale)

struct VisionPayload: Decodable, Equatable {
    let perclos: Double?
    let blinkRate: Double?
    let blinkCount: Int?
    let headPitchMean: Double?
    let headYawMean: Double?
    let headNod: Bool?
    let headDistracted: Bool?
    let yawnCount: Int?
    let yawnActive: Bool?
    let fatigueScore: Double?
    let alertness: String?
    let quality: Double?
    let facePresent: Bool?
    let frames: Int?
    let timestamp: TimeInterval?
    let stale: Bool?

    enum CodingKeys: String, CodingKey {
        case perclos
        case blinkRate = "blink_rate"
        case blinkCount = "blink_count"
        case headPitchMean = "head_pitch_mean"
        case headYawMean = "head_yaw_mean"
        case headNod = "head_nod"
        case headDistracted = "head_distracted"
        case yawnCount = "yawn_count"
        case yawnActive = "yawn_active"
        case fatigueScore = "fatigue_score"
        case alertness, quality
        case facePresent = "face_present"
        case frames, timestamp, stale
    }
}

// MARK: - Stamina

struct StaminaData: Decodable {
    let value: Double
    let state: String
    let consistency: Double
    let tension: Double
    let fatigue: Double
    let drainRate: Double
    let recoveryRate: Double
    let suggestedWorkMin: Double
    let suggestedBreakMin: Double
    let continuousWorkMin: Double
    let totalWorkMin: Double

    enum CodingKeys: String, CodingKey {
        case value, state, consistency, tension, fatigue
        case drainRate = "drain_rate"
        case recoveryRate = "recovery_rate"
        case suggestedWorkMin = "suggested_work_min"
        case suggestedBreakMin = "suggested_break_min"
        case continuousWorkMin = "continuous_work_min"
        case totalWorkMin = "total_work_min"
    }
}

// MARK: - Decision

struct DecisionData: Decodable {
    let state: String
    let recommendation: String
    let urgency: Double
    let reasons: [String]
    let stamina: Double
    let continuousWorkMin: Double
    let totalWorkMin: Double
    let suggestedWorkMin: Double
    let suggestedBreakMin: Double

    enum CodingKeys: String, CodingKey {
        case state, recommendation, urgency, reasons, stamina
        case continuousWorkMin = "continuous_work_min"
        case totalWorkMin = "total_work_min"
        case suggestedWorkMin = "suggested_work_min"
        case suggestedBreakMin = "suggested_break_min"
    }
}

// MARK: - Server Status

struct ServerStatus: Decodable {
    let connected: Bool
    let modelLoaded: Bool
    let demoMode: Bool
    let speed: Double
    let sampleRate: Int
    let channels: Int
    let uptimeSec: Double?
    let websocketClients: Int?

    enum CodingKeys: String, CodingKey {
        case connected
        case modelLoaded = "model_loaded"
        case demoMode = "demo_mode"
        case speed
        case sampleRate = "sample_rate"
        case channels
        case uptimeSec = "uptime_sec"
        case websocketClients = "websocket_clients"
    }
}

// MARK: - Stamina State Enum

enum StaminaState: String, CaseIterable {
    case focused, fading, depleted, recovering

    var displayName: String {
        switch self {
        case .focused:    return "专注"
        case .fading:     return "下降"
        case .depleted:   return "耗尽"
        case .recovering: return "恢复"
        }
    }

    var systemImage: String {
        switch self {
        case .focused:    return "bolt.fill"
        case .fading:     return "bolt.trianglebadge.exclamationmark"
        case .depleted:   return "bolt.slash.fill"
        case .recovering: return "leaf.fill"
        }
    }
}

// MARK: - Recommendation

enum Recommendation: String {
    case keepWorking = "keep_working"
    case takeBreak   = "take_break"
    case startWorking = "start_working"
    case restMore    = "rest_more"

    var displayName: String {
        switch self {
        case .keepWorking:  return "继续工作"
        case .takeBreak:    return "建议休息"
        case .startWorking: return "可以开始"
        case .restMore:     return "继续休息"
        }
    }

    var systemImage: String {
        switch self {
        case .keepWorking:  return "play.fill"
        case .takeBreak:    return "pause.fill"
        case .startWorking: return "arrow.right.circle.fill"
        case .restMore:     return "moon.fill"
        }
    }
}

// MARK: - Connection Mode

enum ConnectionMode: String, CaseIterable, Identifiable {
    case wifi = "WiFi"
    case ble  = "BLE"

    var id: String { rawValue }

    var systemImage: String {
        switch self {
        case .wifi: return "wifi"
        case .ble:  return "antenna.radiowaves.left.and.right"
        }
    }
}
