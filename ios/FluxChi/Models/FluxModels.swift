import Foundation

// MARK: - API Response Envelope

struct FluxResponse<T: Decodable>: Decodable {
    let ok: Bool
    let ts: TimeInterval
    let data: T?
    let error: String?
    let message: String?
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

    enum CodingKeys: String, CodingKey {
        case timestamp, activity, confidence, probabilities, rms
        case emgSampleCount = "emg_sample_count"
        case stamina, decision
    }

    static func == (lhs: FluxState, rhs: FluxState) -> Bool {
        lhs.timestamp == rhs.timestamp
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
