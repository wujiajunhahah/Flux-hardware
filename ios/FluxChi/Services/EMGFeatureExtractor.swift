import Foundation
import CoreML
import Accelerate

/// Extracts 84-dimensional feature vector from raw EMG window for ML inference.
/// 7 features per channel (MAV, RMS, WL, ZC, SSC, MNF, MDF) × 8 channels + 28 pair correlations.
/// For BLE mode（常见 6 路），不足 8 路时零填充；每通道特征在 **去直流** 后计算（对齐 Python `remove_dc`）。
final class EMGFeatureExtractor {

    static let channelCount = 8
    static let featuresPerChannel = 7
    static let pairCount = 28  // C(8,2)
    static let totalFeatures = channelCount * featuresPerChannel + pairCount  // 84

    private static let pairIndices: [(Int, Int)] = {
        var pairs = [(Int, Int)]()
        for i in 0..<channelCount {
            for j in (i + 1)..<channelCount {
                pairs.append((i, j))
            }
        }
        return pairs
    }()

    private let sampleRate: Double
    private let zcThreshold: Double

    /// 默认 320Hz 与 BLE 帧率一致（WAVELETECH WLS128 实测值）
    init(sampleRate: Double = 320, zcThreshold: Double = 20) {
        self.sampleRate = sampleRate
        self.zcThreshold = zcThreshold
    }

    /// Extract 84 features from a multi-channel EMG window.
    /// - Parameter channels: Array of per-channel time series. If < 8 channels, remainder are zero-filled.
    func extract(channels: [[Double]]) -> [Double] {
        var padded = channels
        while padded.count < Self.channelCount {
            let len = channels.first?.count ?? 0
            padded.append([Double](repeating: 0, count: len))
        }

        var features = [Double]()
        features.reserveCapacity(Self.totalFeatures)

        for ch in 0..<Self.channelCount {
            let sig = padded[ch]
            features.append(contentsOf: channelFeatures(sig))
        }

        features.append(contentsOf: pairCorrelations(padded))

        return features
    }

    // MARK: - Per-channel features (7)

    private func removeDC(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return values }
        let m = values.reduce(0, +) / Double(values.count)
        return values.map { $0 - m }
    }

    private func channelFeatures(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [Double](repeating: 0, count: Self.featuresPerChannel) }

        let values = removeDC(values)
        let mav = values.reduce(0) { $0 + abs($1) } / Double(values.count)
        let rms = sqrt(values.reduce(0) { $0 + $1 * $1 } / Double(values.count))
        let wl = zip(values.dropFirst(), values).reduce(0.0) { $0 + abs($1.0 - $1.1) }
        let zc = Double(zeroCrossings(values))
        let ssc = Double(slopeSignChanges(values))
        let (mnf, mdf) = frequencyFeatures(values)

        return [mav, rms, wl, zc, ssc, mnf, mdf]
    }

    private func zeroCrossings(_ v: [Double]) -> Int {
        guard v.count > 1 else { return 0 }
        var count = 0
        for i in 1..<v.count {
            let prev = v[i - 1], curr = v[i]
            guard abs(prev - curr) >= zcThreshold, prev != 0 else { continue }
            if (prev > 0 && curr <= 0) || (prev < 0 && curr >= 0) { count += 1 }
        }
        return count
    }

    private func slopeSignChanges(_ v: [Double]) -> Int {
        guard v.count > 2 else { return 0 }
        var count = 0
        for i in 1..<(v.count - 1) {
            let d1 = v[i] - v[i - 1]
            let d2 = v[i + 1] - v[i]
            guard abs(d1) >= zcThreshold, abs(d2) >= zcThreshold else { continue }
            if d1 * d2 < 0 { count += 1 }
        }
        return count
    }

    /// MNF (mean frequency) and MDF (median frequency) via Accelerate vDSP FFT — O(n log n).
    private func frequencyFeatures(_ signal: [Double]) -> (Double, Double) {
        let n = signal.count
        guard n >= 16 else { return (0, 0) }

        let mean = vDSP.mean(signal)
        var centered = vDSP.add(-mean, signal)
        let energy = vDSP.sumOfSquares(centered)
        guard energy / Double(n) > 1e-6 else { return (0, 0) }

        let log2n = vDSP_Length(log2(Double(n)).rounded(.up))
        let fftN = Int(1 << log2n)
        if centered.count < fftN { centered.append(contentsOf: [Double](repeating: 0, count: fftN - centered.count)) }

        guard let fft = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPDoubleSplitComplex.self) else {
            return (0, 0)
        }

        let nFreq = fftN / 2
        var realp = [Double](repeating: 0, count: nFreq)
        var imagp = [Double](repeating: 0, count: nFreq)
        var psd = [Double](repeating: 0, count: nFreq)

        realp.withUnsafeMutableBufferPointer { rBuf in
            imagp.withUnsafeMutableBufferPointer { iBuf in
                var split = DSPDoubleSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)

                centered.withUnsafeMutableBufferPointer { cBuf in
                    cBuf.baseAddress!.withMemoryRebound(to: DSPDoubleComplex.self, capacity: nFreq) { ptr in
                        vDSP_ctozD(ptr, 2, &split, 1, vDSP_Length(nFreq))
                    }
                }

                fft.transform(input: split, output: &split, direction: .forward)

                vDSP_zvmagsD(&split, 1, &psd, 1, vDSP_Length(nFreq))
            }
        }

        let freqStep = sampleRate / Double(fftN)
        let total = vDSP.sum(psd)
        guard total > 1e-12 else { return (0, 0) }

        var mnfNum = 0.0
        for k in 0..<nFreq { mnfNum += Double(k) * freqStep * psd[k] }
        let mnf = mnfNum / total

        var cum = 0.0
        let half = total / 2
        var mdf = 0.0
        for k in 0..<nFreq {
            cum += psd[k]
            if cum >= half { mdf = Double(k) * freqStep; break }
        }

        return (mnf, mdf)
    }

    // MARK: - Spatial features (28 correlations)

    private func pairCorrelations(_ channels: [[Double]]) -> [Double] {
        let n = channels.first?.count ?? 0
        guard n >= 2 else { return [Double](repeating: 0, count: Self.pairCount) }

        let means = channels.map { ch in ch.reduce(0, +) / Double(ch.count) }
        let stds: [Double] = channels.enumerated().map { i, ch in
            let m = means[i]
            let v = ch.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(ch.count)
            return sqrt(v)
        }

        return Self.pairIndices.map { i, j in
            guard stds[i] > 1e-10, stds[j] > 1e-10 else { return 0 }
            let cov = zip(channels[i], channels[j]).reduce(0.0) {
                $0 + ($1.0 - means[i]) * ($1.1 - means[j])
            } / Double(n)
            return cov / (stds[i] * stds[j])
        }
    }
}

// MARK: - Model Config (从 emg_classifier_config.json 驱动，换模型只需更新 JSON)

/// 模型配置：类别、特征数、输出键名等。
/// 从 bundle 中的 `emg_classifier_config.json` 加载，避免硬编码与模型耦合。
struct EMGClassifierConfig: Codable {
    let classes: [String]
    let n_features: Int
    let window_seconds: Double?
    let channels: Int?

    /// 模型导出时的输出键名（sklearn → coremltools 默认生成 "var_20" 等）。
    /// 若 JSON 未提供，运行时自动探测模型实际输出。
    var output_key: String?

    static let fallbackClasses = ["finger_movement", "rest", "wrist_extend", "wrist_flex", "wrist_movement"]

    static func loadFromBundle() -> EMGClassifierConfig {
        guard let url = Bundle.main.url(forResource: "emg_classifier_config", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let config = try? JSONDecoder().decode(EMGClassifierConfig.self, from: data) else {
            FluxLog.ml.warn("emg_classifier_config.json 未找到或解析失败，使用内置默认值")
            return EMGClassifierConfig(
                classes: fallbackClasses,
                n_features: EMGFeatureExtractor.totalFeatures,
                window_seconds: 0.25,
                channels: 8,
                output_key: nil
            )
        }
        return config
    }
}

// MARK: - CoreML Activity Inference Wrapper

@MainActor
final class EMGActivityInference {

    private var model: MLModel?
    private let config: EMGClassifierConfig
    private let extractor: EMGFeatureExtractor
    /// 运行时解析出的输出键名（首次推理时探测并缓存）
    private var resolvedOutputKey: String?
    private var validationLogged = false

    var classes: [String] { config.classes }

    init() {
        self.config = EMGClassifierConfig.loadFromBundle()
        self.extractor = EMGFeatureExtractor(sampleRate: 320, zcThreshold: 20)
        loadModel()
    }

    struct Prediction {
        let label: String
        let confidence: Double
        let probabilities: [String: Double]
    }

    func predict(channels: [[Double]]) -> Prediction? {
        guard let model else { return nil }
        let features = extractor.extract(channels: channels)

        let nFeatures = NSNumber(value: config.n_features)
        guard let mlArray = try? MLMultiArray(shape: [1, nFeatures], dataType: .float32) else { return nil }
        for (i, v) in features.enumerated() where i < config.n_features {
            mlArray[[0, i] as [NSNumber]] = NSNumber(value: Float(v))
        }

        guard let input = try? MLDictionaryFeatureProvider(
            dictionary: ["features": MLFeatureValue(multiArray: mlArray)]
        ) else { return nil }

        guard let output = try? model.prediction(from: input) else { return nil }

        let outputKey = resolveOutputKey(output: output)
        guard let logits = output.featureValue(for: outputKey)?.multiArrayValue else {
            if !validationLogged {
                validationLogged = true
                let available = output.featureNames.sorted().joined(separator: ", ")
                FluxLog.ml.error("模型输出键 \"\(outputKey)\" 不存在 — 可用键: [\(available)]，推理静默失败")
            }
            return nil
        }

        return softmaxPrediction(logits: logits)
    }

    // MARK: - Output Key Resolution

    /// 首次推理时自动探测模型的输出键名，后续使用缓存值。
    /// 优先级：config.output_key > 自动探测 MultiArray 类型的输出 > "var_20" fallback
    private func resolveOutputKey(output: MLFeatureProvider) -> String {
        if let cached = resolvedOutputKey { return cached }

        if let explicit = config.output_key {
            resolvedOutputKey = explicit
            FluxLog.ml.info("模型输出键: \"\(explicit)\"（config 指定）")
            return explicit
        }

        for name in output.featureNames {
            if let fv = output.featureValue(for: name), fv.multiArrayValue != nil {
                resolvedOutputKey = name
                FluxLog.ml.info("模型输出键: \"\(name)\"（自动探测）— 可用键: \(output.featureNames.sorted())")
                return name
            }
        }

        let fallback = "var_20"
        resolvedOutputKey = fallback
        FluxLog.ml.warn("无法自动探测输出键，回退到 \"\(fallback)\" — 可用键: \(output.featureNames.sorted())")
        return fallback
    }

    // MARK: - Softmax

    private func softmaxPrediction(logits: MLMultiArray) -> Prediction {
        let nClasses = config.classes.count
        var expVals = [Double](repeating: 0, count: nClasses)
        var maxVal = -Double.infinity
        for i in 0..<nClasses {
            let v = logits[i].doubleValue
            if v > maxVal { maxVal = v }
        }
        var sumExp = 0.0
        for i in 0..<nClasses {
            expVals[i] = exp(logits[i].doubleValue - maxVal)
            sumExp += expVals[i]
        }
        let probs = expVals.map { $0 / sumExp }

        var bestIdx = 0
        for i in 1..<probs.count where probs[i] > probs[bestIdx] { bestIdx = i }

        var probDict = [String: Double]()
        for (i, cls) in config.classes.enumerated() { probDict[cls] = probs[i] }

        return Prediction(
            label: config.classes[bestIdx],
            confidence: probs[bestIdx],
            probabilities: probDict
        )
    }

    // MARK: - Model Loading

    private func loadModel() {
        guard let url = Bundle.main.url(forResource: "ActivityClassifier", withExtension: "mlmodelc")
            ?? compileModel() else {
            FluxLog.ml.error("ActivityClassifier 未在 bundle 中找到 (.mlmodelc / .mlpackage)")
            return
        }
        do {
            model = try MLModel(contentsOf: url)
            validateModelContract()
        } catch {
            FluxLog.ml.error("ActivityClassifier 加载失败", error: error)
        }
    }

    /// 启动时一次性校验模型契约：输入/输出维度、类别数
    private func validateModelContract() {
        guard let model else { return }
        let desc = model.modelDescription

        if let inputDesc = desc.inputDescriptionsByName["features"],
           let constraint = inputDesc.multiArrayConstraint {
            let shape = constraint.shape.map(\.intValue)
            let expected = config.n_features
            if !shape.contains(expected) {
                FluxLog.ml.warn("模型输入维度 \(shape) 与 config.n_features=\(expected) 不一致，推理结果可能异常")
            }
        }

        let outputNames = desc.outputDescriptionsByName.keys.sorted()
        FluxLog.ml.info("ActivityClassifier 已加载 — 输出键: \(outputNames), 类别: \(config.classes.count)个, 特征: \(config.n_features)维")
    }

    private func compileModel() -> URL? {
        guard let packageURL = Bundle.main.url(forResource: "ActivityClassifier", withExtension: "mlpackage") else { return nil }
        return try? MLModel.compileModel(at: packageURL)
    }
}
