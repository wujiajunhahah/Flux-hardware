import Foundation
import CoreML

@MainActor
final class PersonalizationManager: ObservableObject {

    @Published var modelVersion: Int = 0
    @Published var trainingCount: Int = 0
    @Published var lastTrainedAt: Date?
    @Published var isTraining = false
    @Published var estimatedAccuracy: Double = 0

    /// Learned offset applied to raw stamina predictions.
    @Published private(set) var calibrationOffset: Double = 0

    private let alpha: Double = 0.3
    private var feedbackPairs: [(predicted: Double, actual: Double)] = []
    private var compiledModelURL: URL?

    private var modelsDir: URL {
        guard let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            fatalError("Documents directory unavailable")
        }
        return base.appendingPathComponent("FluxModels", isDirectory: true)
    }

    private var feedbackDataURL: URL {
        modelsDir.appendingPathComponent("feedback_pairs.json")
    }

    init() {
        try? FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        loadState()
        loadFeedbackPairs()
        compileBaseModel()
    }

    // MARK: - Personalized Prediction

    /// Adjust raw engine stamina with learned calibration.
    func personalizedStamina(_ rawValue: Double) -> Double {
        max(0, min(100, rawValue + calibrationOffset))
    }

    // MARK: - Feedback Collection

    func addTrainingData(session: Session, feedback: UserFeedback) {
        let predicted = session.avgStamina ?? 50
        let actual = feedback.feeling.staminaTarget

        feedbackPairs.append((predicted: predicted, actual: actual))
        trainingCount = feedbackPairs.count

        let error = actual - predicted
        calibrationOffset = calibrationOffset * (1 - alpha) + error * alpha

        let recent = feedbackPairs.suffix(10)
        let avgError = recent.map { abs($0.predicted + calibrationOffset - $0.actual) }
            .reduce(0, +) / Double(recent.count)
        estimatedAccuracy = max(0, min(100, 100 - avgError))

        saveFeedbackPairs()
        saveState()

        if feedbackPairs.count >= 3 && feedbackPairs.count % 3 == 0 {
            Task { await trainCoreMLModel() }
        }
    }

    // MARK: - CoreML (Optional)

    private func compileBaseModel() {
        if let bundled = Bundle.main.url(forResource: "FluxStamina", withExtension: "mlmodelc") {
            compiledModelURL = bundled
            return
        }
        guard let raw = Bundle.main.url(forResource: "FluxStamina", withExtension: "mlmodel") else { return }
        compiledModelURL = try? MLModel.compileModel(at: raw)
    }

    func trainCoreMLModel() async {
        guard let baseURL = activeModelURL() else { return }
        guard feedbackPairs.count >= 3 else { return }

        isTraining = true
        defer { isTraining = false }

        do {
            let batch = try createTrainingBatch()

            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly

            let updateTask = try MLUpdateTask(
                forModelAt: baseURL,
                trainingData: batch,
                configuration: config,
                progressHandlers: MLUpdateProgressHandlers(
                    forEvents: [.trainingBegin, .epochEnd],
                    progressHandler: { _ in },
                    completionHandler: { [weak self] context in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            if context.task.error == nil {
                                let newVersion = self.modelVersion + 1
                                let dest = self.modelsDir
                                    .appendingPathComponent("FluxStamina_v\(newVersion).mlmodelc")
                                try? context.model.write(to: dest)
                                self.modelVersion = newVersion
                                self.lastTrainedAt = Date()
                                self.compiledModelURL = dest
                                self.saveState()
                            }
                        }
                    }
                )
            )
            updateTask.resume()
        } catch {
            print("[ML] Training failed: \(error.localizedDescription)")
        }
    }

    private func createTrainingBatch() throws -> MLBatchProvider {
        let featureCount = 6
        var providers: [MLFeatureProvider] = []

        for pair in feedbackPairs {
            let input = try MLMultiArray(shape: [1, NSNumber(value: featureCount)], dataType: .float32)
            input[0] = NSNumber(value: pair.predicted / 100.0)
            for i in 1..<featureCount { input[i] = 0 }

            let target = try MLMultiArray(shape: [1], dataType: .float32)
            target[0] = NSNumber(value: pair.actual / 100.0)

            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "features": MLFeatureValue(multiArray: input),
                "stamina_target": MLFeatureValue(multiArray: target)
            ])
            providers.append(provider)
        }

        return MLArrayBatchProvider(array: providers)
    }

    private func activeModelURL() -> URL? {
        if modelVersion > 0 {
            let url = modelsDir.appendingPathComponent("FluxStamina_v\(modelVersion).mlmodelc")
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return compiledModelURL
    }

    // MARK: - Persistence

    private func loadState() {
        let d = UserDefaults.standard
        modelVersion = d.integer(forKey: "flux_ml_version")
        trainingCount = d.integer(forKey: "flux_ml_count")
        calibrationOffset = d.double(forKey: "flux_ml_offset")
        estimatedAccuracy = d.double(forKey: "flux_ml_accuracy")
        lastTrainedAt = d.object(forKey: "flux_ml_last_trained") as? Date
    }

    private func saveState() {
        let d = UserDefaults.standard
        d.set(modelVersion, forKey: "flux_ml_version")
        d.set(trainingCount, forKey: "flux_ml_count")
        d.set(calibrationOffset, forKey: "flux_ml_offset")
        d.set(estimatedAccuracy, forKey: "flux_ml_accuracy")
        d.set(lastTrainedAt, forKey: "flux_ml_last_trained")
    }

    private func loadFeedbackPairs() {
        guard let data = try? Data(contentsOf: feedbackDataURL),
              let decoded = try? JSONDecoder().decode([FeedbackPair].self, from: data) else { return }
        feedbackPairs = decoded.map { (predicted: $0.predicted, actual: $0.actual) }
    }

    private func saveFeedbackPairs() {
        let encoded = feedbackPairs.map { FeedbackPair(predicted: $0.predicted, actual: $0.actual) }
        guard let data = try? JSONEncoder().encode(encoded) else { return }
        try? data.write(to: feedbackDataURL)
    }
}

private struct FeedbackPair: Codable {
    let predicted: Double
    let actual: Double
}
