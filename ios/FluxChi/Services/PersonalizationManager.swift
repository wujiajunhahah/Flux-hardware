import Foundation

@MainActor
final class PersonalizationManager: ObservableObject {

    @Published var trainingCount: Int = 0
    @Published var estimatedAccuracy: Double = 0

    /// Learned offset applied to raw stamina predictions.
    @Published private(set) var calibrationOffset: Double = 0

    private let alpha: Double = 0.3
    private var feedbackPairs: [FeedbackPair] = []

    /// WiFi 模式下回传反馈给服务端飞轮
    weak var fluxService: FluxService?

    private var dataDir: URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("FluxModels", isDirectory: true)
    }

    private var feedbackDataURL: URL {
        dataDir.appendingPathComponent("feedback_pairs.json")
    }

    init() {
        try? FileManager.default.createDirectory(at: dataDir, withIntermediateDirectories: true)
        loadState()
        loadFeedbackPairs()
    }

    // MARK: - Personalized Prediction

    func personalizedStamina(_ rawValue: Double) -> Double {
        max(0, min(100, rawValue + calibrationOffset))
    }

    // MARK: - Feedback Collection

    /// 接收 session 反馈，更新标定偏移。同一 session 只学习一次（持久化去重）。
    func addTrainingData(session: Session, feedback: UserFeedback) {
        let sid = session.persistentModelID.hashValue.description
        if feedbackPairs.contains(where: { $0.sessionID == sid }) {
            FluxLog.ml.info("Session \(sid.prefix(8)) 已学习过，跳过重复训练")
            return
        }

        let predicted = session.avgStamina ?? 50
        let actual = feedback.feeling.staminaTarget

        let pair = FeedbackPair(sessionID: sid, predicted: predicted, actual: actual)
        feedbackPairs.append(pair)
        trainingCount = feedbackPairs.count

        let error = actual - predicted
        calibrationOffset = calibrationOffset * (1 - alpha) + error * alpha

        let recent = feedbackPairs.suffix(10)
        let avgError = recent.map { abs($0.predicted + calibrationOffset - $0.actual) }
            .reduce(0, +) / Double(recent.count)
        estimatedAccuracy = max(0, min(100, 100 - avgError))

        saveFeedbackPairs()
        saveState()

        postFeedbackToFlywheel(predicted: predicted, actual: actual)
    }

    // MARK: - Flywheel Integration

    /// 将反馈回传服务端数据飞轮，让全局模型也能从用户反馈中学习
    private func postFeedbackToFlywheel(predicted: Double, actual: Double) {
        guard let service = fluxService else { return }
        let label = actual < 50 ? "fatigued" : "alert"
        let kss: Int
        switch actual {
        case 80...: kss = 2
        case 50..<80: kss = 5
        default: kss = 8
        }

        let url = service.baseURL.appendingPathComponent("api/v1/flywheel/label")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 5
        let body: [String: Any] = [
            "label": label,
            "kss": kss,
            "note": "ios_feedback predicted=\(Int(predicted)) actual=\(Int(actual))"
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        Task.detached {
            _ = try? await URLSession.shared.data(for: request)
        }
    }

    // MARK: - Persistence

    private func loadState() {
        let d = UserDefaults.standard
        trainingCount = d.integer(forKey: "flux_ml_count")
        calibrationOffset = d.double(forKey: "flux_ml_offset")
        estimatedAccuracy = d.double(forKey: "flux_ml_accuracy")
    }

    private func saveState() {
        let d = UserDefaults.standard
        d.set(trainingCount, forKey: "flux_ml_count")
        d.set(calibrationOffset, forKey: "flux_ml_offset")
        d.set(estimatedAccuracy, forKey: "flux_ml_accuracy")
    }

    private func loadFeedbackPairs() {
        guard let data = try? Data(contentsOf: feedbackDataURL),
              let decoded = try? JSONDecoder().decode([FeedbackPair].self, from: data) else { return }
        feedbackPairs = decoded
    }

    private func saveFeedbackPairs() {
        guard let data = try? JSONEncoder().encode(feedbackPairs) else { return }
        try? data.write(to: feedbackDataURL)
    }
}

private struct FeedbackPair: Codable {
    var sessionID: String = ""
    let predicted: Double
    let actual: Double
}
