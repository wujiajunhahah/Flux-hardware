import Foundation

struct PersonalizationProfile: Codable {
    var schemaVersion: Int = 1
    var profileID: String
    var updatedAt: Date
    var trainingCount: Int
    var estimatedAccuracy: Double
    var calibrationOffset: Double
    var feedbackSummary: PersonalizationFeedbackSummary
    var recentFeedbackPairs: [FeedbackPair]
    var deviceCalibrations: [String: DeviceCalibration]
}

struct PersonalizationFeedbackSummary: Codable {
    var totalCount: Int
    var retainedCount: Int
    var avgAbsoluteError: Double
    var lastSessionID: String?
}

struct DeviceCalibration: Codable {
    var deviceID: String
    var deviceName: String
    var calibrationOffset: Double
    var updatedAt: Date
}

struct FeedbackPair: Codable {
    var sessionID: String
    let predicted: Double
    let actual: Double
    var createdAt: Date?

    init(
        sessionID: String = "",
        predicted: Double,
        actual: Double,
        createdAt: Date? = nil
    ) {
        self.sessionID = sessionID
        self.predicted = predicted
        self.actual = actual
        self.createdAt = createdAt
    }

    private enum CodingKeys: String, CodingKey {
        case sessionID, predicted, actual, createdAt
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        sessionID = try container.decodeIfPresent(String.self, forKey: .sessionID) ?? ""
        predicted = try container.decode(Double.self, forKey: .predicted)
        actual = try container.decode(Double.self, forKey: .actual)
        createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt)
    }
}

struct PersonalizationProfileGetData: Decodable {
    let exists: Bool
    let profile: PersonalizationProfile?
}

struct PersonalizationProfilePutData: Decodable {
    let applied: Bool
    let profile: PersonalizationProfile?
}
