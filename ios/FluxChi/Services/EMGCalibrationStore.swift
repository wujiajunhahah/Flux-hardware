import Foundation

/// 每日校准持久化：各通道安静均值与 MVC 峰值，供圆环归一化或后续引擎消费。
struct EMGCalibrationStore: Codable, Equatable {
    var relaxMean: [Double]
    var mvcPeak: [Double]
    /// 0…3 档位质量
    var quality: Int
    var calibratedAt: TimeInterval?

    static let channelCount = 8
    private static let defaultsKey = "flux_emg_calibration_v1"

    static func load() -> EMGCalibrationStore? {
        guard let data = UserDefaults.standard.data(forKey: defaultsKey),
              let v = try? JSONDecoder().decode(EMGCalibrationStore.self, from: data)
        else { return nil }
        return v
    }

    func save() {
        var copy = self
        copy.calibratedAt = Date().timeIntervalSince1970
        guard let data = try? JSONEncoder().encode(copy) else { return }
        UserDefaults.standard.set(data, forKey: Self.defaultsKey)
        UserDefaults.standard.set(copy.calibratedAt, forKey: "flux_last_calibration")
    }

    /// 由采样序列计算质量：用力峰值需明显高于安静。
    static func computeQuality(relax: [Double], mvc: [Double]) -> Int {
        var strong = 0
        for i in 0..<min(relax.count, mvc.count, channelCount) {
            let lo = relax[i]
            let hi = mvc[i]
            if hi > lo * 1.35 + 15 { strong += 1 }
        }
        if strong >= 4 { return 3 }
        if strong >= 2 { return 2 }
        if strong >= 1 { return 1 }
        return 0
    }

    static func empty() -> EMGCalibrationStore {
        EMGCalibrationStore(
            relaxMean: Array(repeating: 0, count: channelCount),
            mvcPeak: Array(repeating: 1, count: channelCount),
            quality: 0,
            calibratedAt: nil
        )
    }

    // MARK: - Aggregates (OnDeviceStaminaEngine / UI)

    /// 8 路安静均值平均，用于续航引擎基线种子。
    var aggregateRelaxMean: Double {
        let n = min(Self.channelCount, relaxMean.count)
        guard n > 0 else { return 0 }
        return relaxMean.prefix(n).reduce(0, +) / Double(n)
    }

    /// 每路 (mvc − relax) 均值，表示个人「用力动态范围」。
    var meanSignalSpan: Double {
        let n = min(min(relaxMean.count, mvcPeak.count), Self.channelCount)
        guard n > 0 else { return 0 }
        var sum = 0.0
        for i in 0..<n {
            sum += max(1, mvcPeak[i] - relaxMean[i])
        }
        return sum / Double(n)
    }

    /// 是否可用于个性化续航（与圆环归一化门槛一致）。
    var isUsableForStamina: Bool {
        meanSignalSpan >= 12 && aggregateRelaxMean >= 1
    }
}
