import Foundation
import CoreML
import Accelerate

/// Extracts 84-dimensional feature vector from raw EMG window for ML inference.
/// 7 features per channel (MAV, RMS, WL, ZC, SSC, MNF, MDF) × 8 channels + 28 pair correlations.
/// For BLE mode (6 channels), channels 7-8 are zero-filled.
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

    init(sampleRate: Double = 1000, zcThreshold: Double = 20) {
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

    private func channelFeatures(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [Double](repeating: 0, count: Self.featuresPerChannel) }

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

// MARK: - CoreML Activity Inference Wrapper

@MainActor
final class EMGActivityInference {

    static let classes = ["finger_movement", "rest", "wrist_extend", "wrist_flex", "wrist_movement"]

    private var model: MLModel?
    private let extractor = EMGFeatureExtractor()

    init() { loadModel() }

    struct Prediction {
        let label: String
        let confidence: Double
        let probabilities: [String: Double]
    }

    /// Run inference on raw EMG channels (6 from BLE → zero-fills to 8).
    func predict(channels: [[Double]]) -> Prediction? {
        guard let model else { return nil }
        let features = extractor.extract(channels: channels)

        guard let mlArray = try? MLMultiArray(shape: [1, 84], dataType: .float32) else { return nil }
        for (i, v) in features.enumerated() {
            mlArray[[0, i] as [NSNumber]] = NSNumber(value: Float(v))
        }

        guard let input = try? MLDictionaryFeatureProvider(
            dictionary: ["features": MLFeatureValue(multiArray: mlArray)]
        ) else { return nil }

        guard let output = try? model.prediction(from: input) else { return nil }

        let rawOutput = output.featureValue(for: "var_20")?.multiArrayValue
        guard let logits = rawOutput else { return nil }

        var expVals = [Double](repeating: 0, count: Self.classes.count)
        var maxVal = -Double.infinity
        for i in 0..<Self.classes.count {
            let v = logits[i].doubleValue
            if v > maxVal { maxVal = v }
        }
        var sumExp = 0.0
        for i in 0..<Self.classes.count {
            expVals[i] = exp(logits[i].doubleValue - maxVal)
            sumExp += expVals[i]
        }
        let probs = expVals.map { $0 / sumExp }

        var bestIdx = 0
        for i in 1..<probs.count {
            if probs[i] > probs[bestIdx] { bestIdx = i }
        }

        var probDict = [String: Double]()
        for (i, cls) in Self.classes.enumerated() { probDict[cls] = probs[i] }

        return Prediction(
            label: Self.classes[bestIdx],
            confidence: probs[bestIdx],
            probabilities: probDict
        )
    }

    private func loadModel() {
        guard let url = Bundle.main.url(forResource: "ActivityClassifier", withExtension: "mlmodelc")
            ?? compileModel() else {
            print("[ML] ActivityClassifier not found in bundle")
            return
        }
        model = try? MLModel(contentsOf: url)
        if model != nil { print("[ML] ActivityClassifier loaded") }
    }

    private func compileModel() -> URL? {
        guard let packageURL = Bundle.main.url(forResource: "ActivityClassifier", withExtension: "mlpackage") else { return nil }
        return try? MLModel.compileModel(at: packageURL)
    }
}
