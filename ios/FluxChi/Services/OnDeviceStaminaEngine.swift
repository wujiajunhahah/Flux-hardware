import Foundation
import Accelerate

/// Three-dimensional stamina engine for real-time EMG processing.
/// D1: Pattern Consistency (signal stability over 30s window)
/// D2: Baseline Tension (current muscle tone vs. learned resting baseline)
/// D3: Sustained Capacity (MDF spectral fatigue indicator)
///
/// Work/rest detection uses RMS variance (movement = variance spike), not ML classifier.
final class OnDeviceStaminaEngine {

    struct Reading {
        let stamina: Double
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
        let isWorking: Bool
    }

    // MARK: - Weights

    private let w1: Double = 0.40
    private let w2: Double = 0.25
    private let w3: Double = 0.35
    private let baseDrain: Double = 1.8
    private let baseRecovery: Double = 5.0

    // MARK: - Adaptive baseline calibration

    private var calibrationSamples: [Double] = []
    private var calibrationVariances: [Double] = []
    private var isCalibrated = false
    private var baselineMeanRMS: Double = 0
    private var baselineVariance: Double = 0
    private let calibrationN = 15

    // MARK: - Work/rest detection via RMS variance

    private var rmsWindow: [Double] = []
    private let varianceWindowSize = 8
    private var activityBuf: [Bool] = []
    private let smoothN = 5

    // MARK: - D1 Consistency (signal stability)

    private var rmsHistory: [(t: Double, v: Double)] = []
    private let consistencyWindow: Double = 30

    // MARK: - D2 Tension (muscle tone vs baseline)

    private var rmsRecent: [Double] = []
    private let tensionWindowN = 300
    private var restBaseline: Double?
    private var baselineBuf: [Double] = []
    private let baselineBufN = 20
    private var tensionEMA: Double = 0

    // MARK: - D3 Fatigue (MDF spectral decline)

    private var mdfHistory: [(t: Double, v: Double)] = []
    private var mdfBaseline: Double?
    private var mdfBaselineBuf: [Double] = []
    private let mdfBaselineN = 15
    private let mdfHistoryLen: Double = 120

    // MARK: - State

    private(set) var stamina: Double = 100
    private var lastTS: Double?
    private var workStart: Double?
    private var totalWorkSec: Double = 0
    private var working = false
    private var drainEMA: Double
    private var recoveryEMA: Double

    init() {
        drainEMA = 1.8
        recoveryEMA = 5.0
    }

    // MARK: - Update

    func update(rms: [Double], rawChannels: [[Double]]?, timestamp: Double, classifiedActivity: String? = nil) -> Reading {
        let activeRMS = rms.filter { $0 > 1 }
        let meanRMS = activeRMS.isEmpty ? 0 : activeRMS.reduce(0, +) / Double(activeRMS.count)

        // --- Adaptive calibration (first ~15 samples = ~7.5 seconds) ---
        if !isCalibrated {
            calibrationSamples.append(meanRMS)
            rmsWindow.append(meanRMS)
            if rmsWindow.count > varianceWindowSize { rmsWindow.removeFirst() }
            if rmsWindow.count >= 4 {
                let v = variance(rmsWindow)
                calibrationVariances.append(v)
            }
            if calibrationSamples.count >= calibrationN {
                baselineMeanRMS = calibrationSamples.reduce(0, +) / Double(calibrationSamples.count)
                baselineVariance = calibrationVariances.isEmpty ? 1 :
                    calibrationVariances.reduce(0, +) / Double(calibrationVariances.count)
                restBaseline = baselineMeanRMS
                isCalibrated = true
            }
        }

        // --- Work/rest detection via RMS variance ---
        rmsWindow.append(meanRMS)
        if rmsWindow.count > varianceWindowSize { rmsWindow.removeFirst() }
        let currentVariance = rmsWindow.count >= 3 ? variance(rmsWindow) : 0
        let varianceThreshold = max(baselineVariance * 3, 5000)
        let rmsDeviation = abs(meanRMS - baselineMeanRMS) / max(baselineMeanRMS, 1)

        let currentWork: Bool
        if let cls = classifiedActivity,
           cls != "rest" && cls != "idle" {
            currentWork = true
        } else {
            currentWork = currentVariance > varianceThreshold || rmsDeviation > 0.3
        }

        activityBuf.append(currentWork)
        if activityBuf.count > smoothN { activityBuf.removeFirst(activityBuf.count - smoothN) }
        let isWork = activityBuf.filter(\.self).count > smoothN / 2

        if isWork && !working {
            working = true
            workStart = timestamp
        } else if !isWork && working {
            if let s = workStart { totalWorkSec += timestamp - s }
            working = false
            workStart = nil
        }

        let contSec = workStart.map { timestamp - $0 } ?? 0
        let totalSec = totalWorkSec + (working ? contSec : 0)

        // --- D1: Pattern Consistency (signal stability, always active) ---
        rmsHistory.append((timestamp, meanRMS))
        rmsHistory.removeAll { timestamp - $0.t > consistencyWindow }
        let con = d1Consistency()

        // --- D2: Tension (muscle tone relative to baseline) ---
        rmsRecent.append(meanRMS)
        if rmsRecent.count > tensionWindowN { rmsRecent.removeFirst() }
        if !isWork && isCalibrated {
            baselineBuf.append(meanRMS)
            if baselineBuf.count > baselineBufN * 2 { baselineBuf.removeFirst() }
            if baselineBuf.count >= baselineBufN {
                let newBaseline = Self.median(baselineBuf)
                restBaseline = restBaseline.map { $0 * 0.9 + newBaseline * 0.1 } ?? newBaseline
            }
        }
        let ten = d2Tension()

        // --- D3: Fatigue (MDF spectral decline) ---
        let fat: Double
        if let ch = rawChannels, !ch.isEmpty, ch[0].count >= 32 {
            // BLE 有效采样率远低于名义 1kHz；用 ~320Hz 尺度减轻 MDF 频偏（与网关估率同量级）
            fat = d3Fatigue(ch, timestamp, sampleRate: 320)
        } else {
            let timeFat = totalSec > 0 ? min(totalSec / 60 / 45, 0.8) : 0
            let varianceFat = isCalibrated && baselineVariance > 0
                ? min(max(0, currentVariance / (baselineVariance * 10) - 0.5), 0.5)
                : 0
            fat = min(1, timeFat + varianceFat)
        }

        // --- Stamina accumulation ---
        var dtMin: Double = 0
        if let last = lastTS { dtMin = max(0, min(timestamp - last, 5)) / 60 }
        lastTS = timestamp

        if isWork {
            let fatMul = 0.5 + fat
            var drain = baseDrain * fatMul * dtMin
            drain += max(0, 0.5 - con) * 0.5 * dtMin
            drain += ten * 0.3 * dtMin
            stamina = max(0, stamina - drain)
            if dtMin > 1e-9 { drainEMA = 0.9 * drainEMA + 0.1 * (drain / dtMin) }
        } else {
            let quality = max(0.1, 1 - ten)
            let rec = baseRecovery * quality * dtMin
            stamina = min(100, stamina + rec)
            if dtMin > 1e-9 { recoveryEMA = 0.9 * recoveryEMA + 0.1 * (rec / dtMin) }
        }

        let state: String = {
            if !isWork { return "recovering" }
            if stamina > 60 { return "focused" }
            if stamina > 30 { return "fading" }
            return "depleted"
        }()

        let dr = max(0.01, drainEMA)
        let rr = max(0.01, recoveryEMA)
        let sugWork = isWork ? max(0, (stamina - 30) / dr) : max(0, (stamina - 30) / dr)
        let sugBreak: Double = {
            let deficit = 70 - stamina
            return deficit > 0 ? deficit / rr : 0
        }()

        return Reading(
            stamina: (stamina * 10).rounded() / 10,
            state: state,
            consistency: (con * 1000).rounded() / 1000,
            tension: (ten * 1000).rounded() / 1000,
            fatigue: (fat * 1000).rounded() / 1000,
            drainRate: (dr * 100).rounded() / 100,
            recoveryRate: (rr * 100).rounded() / 100,
            suggestedWorkMin: (sugWork * 10).rounded() / 10,
            suggestedBreakMin: (sugBreak * 10).rounded() / 10,
            continuousWorkMin: (contSec / 60 * 10).rounded() / 10,
            totalWorkMin: (totalSec / 60 * 10).rounded() / 10,
            isWorking: isWork
        )
    }

    func reset() {
        stamina = 100; lastTS = nil; workStart = nil; totalWorkSec = 0
        rmsHistory.removeAll(); rmsRecent.removeAll(); activityBuf.removeAll()
        mdfHistory.removeAll(); restBaseline = nil; baselineBuf.removeAll()
        mdfBaseline = nil; mdfBaselineBuf.removeAll(); working = false
        drainEMA = baseDrain; recoveryEMA = baseRecovery
        calibrationSamples.removeAll(); calibrationVariances.removeAll()
        isCalibrated = false; baselineMeanRMS = 0; baselineVariance = 0
        rmsWindow.removeAll(); tensionEMA = 0
    }

    // MARK: - D1 Consistency (always active, measures signal stability)

    private func d1Consistency() -> Double {
        guard rmsHistory.count >= 5 else { return 0 }
        let vals = rmsHistory.map(\.v)
        let mean = vals.reduce(0, +) / Double(vals.count)
        guard mean > 1e-6 else { return 0 }
        let std = sqrt(vals.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(vals.count))
        let cv = std / mean
        return max(0, min(1, 1 - cv))
    }

    // MARK: - D2 Tension (current muscle tone vs resting baseline)

    private func d2Tension() -> Double {
        guard rmsRecent.count >= 5 else { return 0 }

        let sorted = rmsRecent.sorted()
        let p10Idx = max(0, Int(Double(sorted.count) * 0.1))
        let p10 = sorted[p10Idx]
        let p90 = sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.9))]

        let base = restBaseline ?? baselineMeanRMS
        guard base > 10 else {
            let range = p90 - p10
            let mean = rmsRecent.reduce(0, +) / Double(rmsRecent.count)
            return mean > 10 ? min(1, range / mean) : 0
        }

        let deviation = max(0, (p10 - base) / base)
        let rangeRatio = (p90 - p10) / max(base, 1)
        let raw = deviation * 0.6 + rangeRatio * 0.4
        tensionEMA = tensionEMA * 0.85 + raw * 0.15
        return max(0, min(1, tensionEMA))
    }

    // MARK: - D3 Fatigue (MDF spectral decline)

    private func d3Fatigue(_ channels: [[Double]], _ ts: Double, sampleRate: Double) -> Double {
        var mdfs = [Double]()
        for ch in channels {
            guard ch.count >= 32 else { continue }
            let mdf = Self.fftMDF(ch, sampleRate: sampleRate)
            if mdf > 0 { mdfs.append(mdf) }
        }
        guard !mdfs.isEmpty else { return 0 }
        let currentMDF = Self.median(mdfs)

        if mdfBaseline == nil {
            mdfBaselineBuf.append(currentMDF)
            if mdfBaselineBuf.count >= mdfBaselineN {
                mdfBaseline = Self.median(mdfBaselineBuf)
                mdfBaselineBuf.removeAll()
            }
        }

        mdfHistory.append((ts, currentMDF))
        mdfHistory.removeAll { ts - $0.t > mdfHistoryLen }

        guard let baseline = mdfBaseline, baseline > 1e-6 else { return 0 }
        let drop = max(0, (baseline - currentMDF) / baseline)
        let slope = mdfSlope()
        let slopeTerm = max(0, -slope / (baseline + 1e-6))
        return max(0, min(1, 0.6 * drop + 0.4 * min(slopeTerm * 60, 1)))
    }

    // MARK: - Helpers

    private func variance(_ values: [Double]) -> Double {
        guard values.count >= 2 else { return 0 }
        let mean = values.reduce(0, +) / Double(values.count)
        return values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count)
    }

    private func mdfSlope() -> Double {
        guard mdfHistory.count >= 3 else { return 0 }
        let t0 = mdfHistory[0].t
        let xs = mdfHistory.map { $0.t - t0 }
        let ys = mdfHistory.map(\.v)
        let n = Double(xs.count)
        let sx = xs.reduce(0, +), sy = ys.reduce(0, +)
        let sxy = zip(xs, ys).map(*).reduce(0, +)
        let sxx = xs.map { $0 * $0 }.reduce(0, +)
        let d = n * sxx - sx * sx
        guard abs(d) > 1e-12 else { return 0 }
        return (n * sxy - sx * sy) / d
    }

    /// Median Frequency via Accelerate vDSP FFT — O(n log n).
    static func fftMDF(_ signal: [Double], sampleRate: Double) -> Double {
        let n = signal.count
        guard n >= 16 else { return 0 }

        let mean = vDSP.mean(signal)
        var centered = vDSP.add(-mean, signal)
        let energy = vDSP.sumOfSquares(centered)
        guard energy / Double(n) > 1e-6 else { return 0 }

        let log2n = vDSP_Length(log2(Double(n)).rounded(.up))
        let fftN = Int(1 << log2n)
        if centered.count < fftN {
            centered.append(contentsOf: [Double](repeating: 0, count: fftN - centered.count))
        }

        guard let fft = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPDoubleSplitComplex.self) else {
            return 0
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

        let total = vDSP.sum(psd)
        guard total > 1e-12 else { return 0 }
        let half = total / 2
        let step = sampleRate / Double(fftN)
        var cum = 0.0
        for (i, p) in psd.enumerated() {
            cum += p
            if cum >= half { return Double(i) * step }
        }
        return 0
    }

    static func median(_ values: [Double]) -> Double {
        let s = values.sorted()
        guard !s.isEmpty else { return 0 }
        return s.count % 2 == 0 ? (s[s.count / 2 - 1] + s[s.count / 2]) / 2 : s[s.count / 2]
    }
}
