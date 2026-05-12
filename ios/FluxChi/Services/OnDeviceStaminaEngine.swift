import Foundation
import Accelerate

/// Three-dimensional stamina engine for real-time EMG processing.
/// D1: Pattern Consistency (signal stability over 30s window)
/// D2: Baseline Tension (current muscle tone vs. learned resting baseline)
/// D3: Sustained Capacity (MDF spectral fatigue indicator)
///
/// Work/rest detection uses RMS variance (movement = variance spike), not ML classifier.
///
/// Actor 隔离：所有可变状态（rmsHistory、mdfHistory、calibration 缓冲等）由 actor 串行化，
/// 调用方 `await update(...)` 时 vDSP FFT 与累积器运算在 actor 域执行，自动离开 MainActor。
actor OnDeviceStaminaEngine {

    struct Reading: Sendable {
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

    // MARK: - Weights (引用 Flux.StaminaWeights — 单一权重源)

    private let w1: Double = Flux.StaminaWeights.consistency
    private let w2: Double = Flux.StaminaWeights.tension
    private let w3: Double = Flux.StaminaWeights.fatigue
    private let baseDrain: Double = Flux.StaminaWeights.baseDrain
    private let baseRecovery: Double = Flux.StaminaWeights.baseRecovery

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

    // MARK: - D3 Fatigue (multi-domain: MDF + low-band ratio + RMS trend)

    private var mdfHistory: [(t: Double, v: Double)] = []
    private var mdfBaseline: Double?
    private var mdfBaselineBuf: [Double] = []
    private let mdfBaselineN = 15
    private let mdfHistoryLen: Double = 120

    /// 低频能量比（20-60Hz / 20-200Hz）的 baseline。疲劳时此值上升。
    /// 与 MDF 互补：MDF 对量纲不变化敏感、lowBand 对绝对量级偏移更鲁棒。
    private var lowBandBaseline: Double?
    private var lowBandBaselineBuf: [Double] = []

    /// 进入 rest 状态的时刻；rest 持续 > rebaseliningRestSec 时触发 MDF baseline 重建，
    /// 解决长时间使用后电极位移导致 baseline 漂移、被误识别成「持续疲劳」的问题。
    private var restStartedAt: Double?
    private let rebaseliningRestSec: Double = 300  // 5 分钟连续 rest 触发

    /// 工作期间 RMS 的滚动均值，用于检测「持续工作中 RMS 缓慢上升 = 疲劳代偿」模式。
    /// 力竭时为了维持输出，更多运动单元参与，RMS 增加。
    private var workRMSHistory: [(t: Double, v: Double)] = []
    private let workRMSHistoryLen: Double = 180  // 3 min 滚动窗

    // MARK: - State

    private(set) var stamina: Double = 100
    private var lastTS: Double?
    private var workStart: Double?
    private var totalWorkSec: Double = 0
    private var working = false
    private var drainEMA: Double
    private var recoveryEMA: Double

    init() {
        drainEMA = Flux.StaminaWeights.baseDrain
        recoveryEMA = Flux.StaminaWeights.baseRecovery
    }

    // MARK: - Update

    func update(rms: [Double], rawChannels: [[Double]]?, timestamp: Double, classifiedActivity: String? = nil, sampleRateHz: Double = 320, imuMotion: Double = 0) -> Reading {
        let profile = EMGCalibrationStore.load()
        let useProfile = profile?.isUsableForStamina == true
        let targetCalibrationN = useProfile ? 8 : calibrationN
        let personalSpan = max(profile?.meanSignalSpan ?? 25, 25)

        // 通道质量加权聚合：电极接触不良时单路 RMS 会比 baseline 高 50-200×，
        // 直接 mean(rms) 会被噪声电极主导。按每路相对 baseline 的合理度加权。
        // 无 calibration 时回退到 RMS>1 过滤的简单均值。
        let meanRMS = Self.weightedMeanRMS(rms: rms, profile: profile)

        // --- Adaptive calibration（无档案约 15 帧；有每日校准档案约 8 帧并融合安静基线）---
        if !isCalibrated {
            calibrationSamples.append(meanRMS)
            rmsWindow.append(meanRMS)
            if rmsWindow.count > varianceWindowSize { rmsWindow.removeFirst() }
            if rmsWindow.count >= 4 {
                let v = variance(rmsWindow)
                calibrationVariances.append(v)
            }
            if calibrationSamples.count >= targetCalibrationN {
                let onlineMean = calibrationSamples.reduce(0, +) / Double(calibrationSamples.count)
                if useProfile, let p = profile {
                    baselineMeanRMS = onlineMean * 0.35 + p.aggregateRelaxMean * 0.65
                } else {
                    baselineMeanRMS = onlineMean
                }
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
        let rmsExcursion = abs(meanRMS - baselineMeanRMS) / max(personalSpan, 1)

        // IMU 增强 work/rest 判定：手在动（gyro magnitude > 0.3 rad/s ≈ 17°/s）几乎肯定不是 rest，
        // 这能避免「手在打字但 EMG 噪声低 → 误判 rest」的情况。
        let imuIndicatesActivity = imuMotion > 0.3

        let currentWork: Bool
        if let cls = classifiedActivity,
           cls != "rest" && cls != "idle" {
            currentWork = true
        } else if imuIndicatesActivity {
            currentWork = true
        } else {
            let strongDeviation = useProfile ? (rmsExcursion > 0.26) : (rmsDeviation > 0.3)
            currentWork = currentVariance > varianceThreshold || strongDeviation
        }

        activityBuf.append(currentWork)
        if activityBuf.count > smoothN { activityBuf.removeFirst(activityBuf.count - smoothN) }
        let isWork = activityBuf.filter(\.self).count > smoothN / 2

        if isWork && !working {
            working = true
            workStart = timestamp
            restStartedAt = nil  // 工作开始，重置 rest 计时
        } else if !isWork && working {
            if let s = workStart { totalWorkSec += timestamp - s }
            working = false
            workStart = nil
            restStartedAt = timestamp  // 进入 rest，开始计时
        } else if !isWork && restStartedAt == nil {
            // 系统刚启动就处于 rest，也记录起点
            restStartedAt = timestamp
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

        // --- D3: Fatigue (multi-domain: MDF + low-band + RMS rise) ---
        let fat: Double
        if let ch = rawChannels, !ch.isEmpty, ch[0].count >= 32 {
            // 使用 BLEManager 实测的 fps，而非硬编码 320Hz。
            // 之前用错的采样率会让 MDF 频率值整体偏移 3× 以上，物理意义错乱。
            let rawFat = d3Fatigue(ch, timestamp, sampleRate: sampleRateHz, meanRMS: meanRMS, isWork: isWork)
            // IMU 抑制：剧烈运动（>1.5 rad/s ≈ 大幅挥手）下 MDF 受运动伪迹污染，可信度下降。
            // 线性缩放：motion=0 → ×1.0, motion=1.5 → ×0.5, motion≥3 → ×0.2
            let imuConfidence = max(0.2, 1.0 - imuMotion / 3.0)
            fat = rawFat * imuConfidence
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
            if !isWork { return StaminaState.recovering.rawValue }
            if stamina > Flux.StaminaWeights.focusedMin { return StaminaState.focused.rawValue }
            if stamina > Flux.StaminaWeights.fadingMin  { return StaminaState.fading.rawValue }
            return StaminaState.depleted.rawValue
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
        restStartedAt = nil
        lowBandBaseline = nil; lowBandBaselineBuf.removeAll()
        workRMSHistory.removeAll()
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

    // MARK: - D2 Tension (current muscle tone vs resting baseline, adaptive EMA)

    /// 张力估计 — 用自适应 EMA + baseline 离散度做 z-score 风格的归一化。
    ///
    /// 改进点：
    /// 1. **Baseline std 替代固定 EMA α**：从 baselineBuf 算出 std，让"显著偏离"用 σ 单位
    ///    而非绝对 RMS 值表达 — 跨用户/跨 session 一致。
    /// 2. **自适应 EMA**：变化大时 α=0.30 快跟踪（捕捉紧张突发），稳态时 α=0.10 抗噪。
    ///    旧版固定 α=0.15 在两个极端都次优。
    /// 3. **保留 p10/p90 percentile** 思路（抗瞬时尖峰）。
    private func d2Tension() -> Double {
        guard rmsRecent.count >= 5 else { return 0 }

        let sorted = rmsRecent.sorted()
        let p10Idx = max(0, Int(Double(sorted.count) * 0.1))
        let p10 = sorted[p10Idx]
        let p90 = sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.9))]

        let base = restBaseline ?? baselineMeanRMS
        guard base > 10 else {
            // 无 baseline：回退到范围 / 均值
            let range = p90 - p10
            let mean = rmsRecent.reduce(0, +) / Double(rmsRecent.count)
            return mean > 10 ? min(1, range / mean) : 0
        }

        // 用 baselineBuf 的 std 做归一化 — 跨人/跨 session 在统计意义上一致
        // baselineBuf 可能未满，给个 floor 避免 σ=0
        let baselineStd: Double = {
            guard baselineBuf.count >= 4 else { return max(base * 0.10, 5.0) }  // 默认 σ ≈ 10% baseline
            return max(variance(baselineBuf).squareRoot(), base * 0.05)
        }()

        // 用 std 而非绝对值的相对偏离：1.0 = 偏离 1σ
        let deviationSigma = max(0, (p10 - base) / baselineStd)
        let rangeSigma = (p90 - p10) / baselineStd
        // 归一化到 [0, 1] —— 3σ 视为接近满量程
        let raw = min(1.0, (deviationSigma * 0.6 + rangeSigma * 0.4) / 3.0)

        // 自适应 EMA：根据当前 raw 与 ema 的差异决定 α
        let delta = abs(raw - tensionEMA)
        let alpha: Double
        if delta > 0.15 { alpha = 0.30 }       // 快跟（突变）
        else if delta > 0.05 { alpha = 0.20 }  // 中速
        else { alpha = 0.10 }                  // 慢跟（抗噪）
        tensionEMA = tensionEMA * (1 - alpha) + raw * alpha

        return max(0, min(1, tensionEMA))
    }

    // MARK: - D3 Fatigue (MDF spectral decline)

    /// 多域 fatigue 融合（取代旧的单 MDF 实现）。
    /// 三个 indicator + 加权融合（参考 MDPI Wireless EMG Fatigue Index 2025）：
    ///   - mdfDrop:        MDF 相对 baseline 的下降比例（频谱中位左移）
    ///   - lowBandIncrease: 20-60Hz 能量比相对 baseline 的上升（频谱低频压缩）
    ///   - rmsRise:        工作期间 RMS 的上升（代偿性招募，力竭信号）
    /// 三个 signal 在物理上互补，融合后比单一 MDF 鲁棒得多。
    private func d3Fatigue(_ channels: [[Double]], _ ts: Double, sampleRate: Double, meanRMS: Double, isWork: Bool) -> Double {
        // 逐通道算 MDF + 低频比，取 median 减少噪声
        var mdfs = [Double](), lowBands = [Double]()
        for ch in channels {
            guard ch.count >= 32 else { continue }
            let (mdf, ratio) = Self.fftMDFAndLowBand(ch, sampleRate: sampleRate)
            if mdf > 0 {
                mdfs.append(mdf)
                lowBands.append(ratio)
            }
        }
        guard !mdfs.isEmpty else { return 0 }
        let currentMDF = Self.median(mdfs)
        let currentLowBand = Self.median(lowBands)

        // --- Baseline 建立（MDF & lowBand 同步） ---
        if mdfBaseline == nil {
            mdfBaselineBuf.append(currentMDF)
            lowBandBaselineBuf.append(currentLowBand)
            if mdfBaselineBuf.count >= mdfBaselineN {
                mdfBaseline = Self.median(mdfBaselineBuf)
                lowBandBaseline = Self.median(lowBandBaselineBuf)
                mdfBaselineBuf.removeAll()
                lowBandBaselineBuf.removeAll()
            }
        } else if let restStart = restStartedAt,
                  ts - restStart > rebaseliningRestSec,
                  !isWork {
            // 长 rest 触发 baseline 重建（电极漂移补偿）
            let recentMDFs = mdfHistory.filter { ts - $0.t < rebaseliningRestSec }.map(\.v)
            if recentMDFs.count >= 5 {
                mdfBaseline = Self.median(recentMDFs)
                lowBandBaseline = currentLowBand  // lowBand 单点替换够用，因为它本身较稳定
                restStartedAt = ts
                FluxLog.ml.info("Fatigue baseline re-rotated: MDF=\(String(format: "%.1f", mdfBaseline ?? 0))Hz, lowBand=\(String(format: "%.2f", lowBandBaseline ?? 0))")
            }
        }

        // 历史追踪
        mdfHistory.append((ts, currentMDF))
        mdfHistory.removeAll { ts - $0.t > mdfHistoryLen }
        if isWork {
            workRMSHistory.append((ts, meanRMS))
            workRMSHistory.removeAll { ts - $0.t > workRMSHistoryLen }
        }

        guard let mdfBase = mdfBaseline, mdfBase > 1e-6 else { return 0 }

        // --- Indicator 1: MDF drop（频谱中位左移） ---
        let mdfDrop = max(0, (mdfBase - currentMDF) / mdfBase)

        // --- Indicator 2: MDF slope 趋势 ---
        let slope = mdfSlope()
        let slopeTerm = max(0, -slope / (mdfBase + 1e-6))

        // --- Indicator 3: Low-band ratio 增长 ---
        let lowBandTerm: Double
        if let lbBase = lowBandBaseline, lbBase > 0.05 {
            lowBandTerm = max(0, min(1, (currentLowBand - lbBase) / max(lbBase, 0.1)))
        } else {
            lowBandTerm = 0
        }

        // --- Indicator 4: 工作期间 RMS 缓慢上升（运动单元代偿招募） ---
        let rmsRiseTerm = workRMSRiseRatio()

        // --- 加权融合 ---
        // 权重设计：MDF drop 仍是主信号（最 well-validated），低频比第二，slope 第三，RMS rise 辅助。
        // 总权重和为 1.0。
        let fatigue = 0.40 * mdfDrop
                    + 0.25 * lowBandTerm
                    + 0.20 * min(slopeTerm * 60, 1)
                    + 0.15 * rmsRiseTerm

        return max(0, min(1, fatigue))
    }

    /// 工作期间 RMS 滚动均值的上升趋势归一化到 [0, 1]。
    /// 力竭代偿时 RMS 缓慢上升（更多运动单元招募），健康疲劳信号。
    private func workRMSRiseRatio() -> Double {
        guard workRMSHistory.count >= 6 else { return 0 }
        // 用前 1/3 vs 后 1/3 的均值比较，比单纯 slope 鲁棒
        let n = workRMSHistory.count
        let third = max(2, n / 3)
        let earlyMean = workRMSHistory.prefix(third).map(\.v).reduce(0, +) / Double(third)
        let lateMean = workRMSHistory.suffix(third).map(\.v).reduce(0, +) / Double(third)
        guard earlyMean > 10 else { return 0 }
        let rise = (lateMean - earlyMean) / earlyMean
        // 涨 20%+ 算明显疲劳代偿
        return max(0, min(1, rise / 0.20))
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

    /// 单通道频谱分析。返回 `(MDF, lowBandRatio)`。
    /// - MDF: 中位频率（Hz），疲劳时下降。
    /// - lowBandRatio: 20-60Hz 能量 / 20-200Hz 能量，疲劳时上升（频谱左移）。
    /// 两者互补 — MDF 抗个体差异，lowBandRatio 抗 baseline 漂移。
    static func fftMDFAndLowBand(_ signal: [Double], sampleRate: Double) -> (mdf: Double, lowBandRatio: Double) {
        let n = signal.count
        guard n >= 16, sampleRate > 0 else { return (0, 0) }

        let mean = vDSP.mean(signal)
        var centered = vDSP.add(-mean, signal)
        let energy = vDSP.sumOfSquares(centered)
        guard energy / Double(n) > 1e-6 else { return (0, 0) }

        let log2n = vDSP_Length(log2(Double(n)).rounded(.up))
        let fftN = Int(1 << log2n)
        if centered.count < fftN {
            centered.append(contentsOf: [Double](repeating: 0, count: fftN - centered.count))
        }

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

        let step = sampleRate / Double(fftN)
        let total = vDSP.sum(psd)
        guard total > 1e-12 else { return (0, 0) }

        // MDF：累积能量超过一半时的频率
        let half = total / 2
        var cum = 0.0
        var mdf = 0.0
        for (i, p) in psd.enumerated() {
            cum += p
            if cum >= half { mdf = Double(i) * step; break }
        }

        // 低/总能量比：[20, 60] Hz / [20, 200] Hz
        // sEMG 主要能量 20-200Hz，疲劳时谱包络向低频压缩
        var lowSum = 0.0, totalBandSum = 0.0
        for (i, p) in psd.enumerated() {
            let freq = Double(i) * step
            if freq >= 20 && freq <= 200 {
                totalBandSum += p
                if freq <= 60 { lowSum += p }
            }
        }
        let ratio = totalBandSum > 1e-12 ? lowSum / totalBandSum : 0

        return (mdf, ratio)
    }

    /// 旧 API 保持兼容（仅返回 MDF），内部走新实现。
    static func fftMDF(_ signal: [Double], sampleRate: Double) -> Double {
        fftMDFAndLowBand(signal, sampleRate: sampleRate).mdf
    }

    /// Legacy 实现（保留用于潜在 fallback）
    static func fftMDF_legacy(_ signal: [Double], sampleRate: Double) -> Double {
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

    /// 通道质量加权 RMS 均值。
    /// 当存在每日校准时，每路 ratio = rms[i] / baseline[i] 应在 [0.5, 10] 内为正常活动。
    /// 超过 10× 几乎肯定是电极故障，降权；超过 30× 几乎完全忽略。
    /// 这样一个掉线的电极不会让全 8 路均值飘到天上去。
    static func weightedMeanRMS(rms: [Double], profile: EMGCalibrationStore?) -> Double {
        guard let baseline = profile?.relaxMean,
              baseline.count >= rms.count else {
            // 无 calibration：回退到原本的「过滤静音通道」逻辑
            let active = rms.filter { $0 > 1 }
            return active.isEmpty ? 0 : active.reduce(0, +) / Double(active.count)
        }

        var totalW = 0.0
        var weightedSum = 0.0
        for i in 0..<rms.count {
            let base = max(baseline[i], 5.0)  // floor 避免 baseline=0 时除零
            let ratio = rms[i] / base
            let w: Double
            if ratio < 0.3 { w = 0.0 }       // 通道死了
            else if ratio < 10 { w = 1.0 }    // 正常范围
            else if ratio < 30 { w = 0.4 }    // 强信号但可能电极松了
            else { w = 0.05 }                 // 几乎肯定噪声
            weightedSum += rms[i] * w
            totalW += w
        }
        guard totalW > 0 else { return 0 }
        return weightedSum / totalW
    }
}
