import Foundation

// MARK: - Defaults Keys (集中所有 UserDefaults 字符串)

extension Flux {
    /// 所有 `UserDefaults` 键的单一来源。新增键必须加在这里，禁止裸字面量。
    enum DefaultsKeys {
        // App 级
        static let onboardingDone     = "flux_onboarding_done"
        static let lastCalibration    = "flux_last_calibration"

        // Endpoint
        static let host               = "flux_host"
        static let port               = "flux_port"

        // 个性化
        static let mlCount            = "flux_ml_count"
        static let mlOffset           = "flux_ml_offset"
        static let mlAccuracy         = "flux_ml_accuracy"
        static let mlProfileID        = "flux_ml_profile_id"
        static let mlDeviceID         = "flux_ml_device_id"
        static let mlProfileUpdatedAt = "flux_ml_profile_updated_at"
        static let mlLastSyncAt       = "flux_ml_last_sync_at"
        static let mlDeviceCalibrations = "flux_ml_device_calibrations"

        // EMG 校准
        static let emgCalibrationV1   = "flux_emg_calibration_v1"

        // Platform Auth
        static let platformClientDeviceKey = "flux_platform_client_device_key"
        static let platformSession    = "flux_platform_session"
    }

    /// `UserDefaults` 上的便捷 helper，避免在 View / Service 层重复读相同 key。
    enum Calibration {
        /// 今天是否完成过每日校准（与 `EMGCalibrationStore.save` 写入的 `lastCalibration` 时间戳对齐）。
        static var isCalibratedToday: Bool {
            let last = UserDefaults.standard.double(forKey: DefaultsKeys.lastCalibration)
            guard last > 0 else { return false }
            return Calendar.current.isDateInToday(Date(timeIntervalSince1970: last))
        }
    }
}

// MARK: - Time Slot (上午/午间/下午/晚间)

extension Flux {
    /// 工作日时段切分。统一被 Dashboard / WidgetDataManager / BodyInsightEngine / TimeSlotChartCard 复用。
    enum TimeSlot: String, CaseIterable {
        case morning = "上午"
        case noon    = "午间"
        case afternoon = "下午"
        case evening = "晚间"
        case other   = "其他"

        /// 6-12 上午, 12-14 午间, 14-18 下午, 18-22 晚间, 其余其他。
        static func from(hour: Int) -> TimeSlot {
            switch hour {
            case 6..<12:  return .morning
            case 12..<14: return .noon
            case 14..<18: return .afternoon
            case 18..<22: return .evening
            default:      return .other
            }
        }

        static func from(date: Date, calendar: Calendar = .current) -> TimeSlot {
            from(hour: calendar.component(.hour, from: date))
        }

        /// 排序用。
        var order: Int {
            switch self {
            case .morning:   return 0
            case .noon:      return 1
            case .afternoon: return 2
            case .evening:   return 3
            case .other:     return 4
            }
        }

        var iconName: String {
            switch self {
            case .morning:   return "sunrise.fill"
            case .noon:      return "sun.max.fill"
            case .afternoon: return "sun.haze.fill"
            case .evening:   return "moon.stars.fill"
            case .other:     return "clock.fill"
            }
        }
    }
}

// MARK: - Stamina Weights (续航三维权重)

extension Flux {
    /// 三维续航权重的单一来源。任何要修改 0.40/0.25/0.35 的地方都必须改这里。
    /// 同步参考：`src/energy.py` (Python 端 StaminaEngine)、论文表 §3.2.2。
    enum StaminaWeights {
        static let consistency: Double = 0.40
        static let tension: Double     = 0.25
        static let fatigue: Double     = 0.35

        static let baseDrain: Double    = 1.8
        static let baseRecovery: Double = 5.0

        /// 状态阈值
        static let focusedMin: Double = 60
        static let fadingMin: Double  = 30

        /// JSON 导出用字典 — `ExportManager` 引用本字典而非自己写一遍。
        static var weightsDict: [String: Double] {
            ["consistency": consistency, "tension": tension, "fatigue": fatigue]
        }

        static var thresholdsDict: [String: Double] {
            ["focused": focusedMin, "fading": fadingMin, "depleted": 0]
        }
    }
}

// MARK: - Retry / Backoff Policy

extension Flux {
    /// 网络退避策略，被 SSE/轮询/feedback-event 上传共用。
    enum RetryPolicy {
        /// 起始间隔 0.5s，指数 ×2，封顶 5s。
        static let pollMinDelayMs: Int = 500
        static let pollMaxDelayMs: Int = 5_000

        /// 计算下一次轮询/重连的等待毫秒数。`failureCount=0` 即首次成功后立即恢复 500ms 节奏。
        static func nextDelayMs(failureCount: Int) -> Int {
            guard failureCount > 0 else { return pollMinDelayMs }
            let scaled = pollMinDelayMs * (1 << min(failureCount - 1, 4))
            // 加 0–25% jitter，避免雪崩
            let jitter = Int.random(in: 0...(scaled / 4))
            return min(pollMaxDelayMs, scaled + jitter)
        }
    }
}
