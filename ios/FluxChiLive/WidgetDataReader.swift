import Foundation

/// Widget Extension 端的数据读取层
/// 与主 App 的 WidgetDataManager 共享 App Group UserDefaults key
enum WidgetDataManager {

    static let appGroupID = "group.com.fluxchi.app"

    // MARK: - Data Model

    struct WidgetSnapshot: Codable {
        let todaySessionCount: Int
        let todayTotalMin: Int
        let todayAvgStamina: Double
        let bestSlotName: String?
        let bestSlotAvg: Double
        let lastSessionTime: Date?
        let lastSessionStamina: Double
        let weeklyAvgStamina: Double
        let weeklyDays: [DayEntry]
        let lastUpdated: Date

        struct DayEntry: Codable, Identifiable {
            var id: Date { date }
            let date: Date
            let avgStamina: Double
            let sessionCount: Int
        }

        static let empty = WidgetSnapshot(
            todaySessionCount: 0,
            todayTotalMin: 0,
            todayAvgStamina: 0,
            bestSlotName: nil,
            bestSlotAvg: 0,
            lastSessionTime: nil,
            lastSessionStamina: 0,
            weeklyAvgStamina: 0,
            weeklyDays: [],
            lastUpdated: Date()
        )
    }

    // MARK: - Keys (与主 App 一致)

    private enum Key {
        static let todaySessionCount = "widget_today_session_count"
        static let todayTotalMin = "widget_today_total_min"
        static let todayAvgStamina = "widget_today_avg_stamina"
        static let bestSlotName = "widget_best_slot_name"
        static let bestSlotAvg = "widget_best_slot_avg"
        static let lastSessionTime = "widget_last_session_time"
        static let lastSessionStamina = "widget_last_session_stamina"
        static let weeklyAvgStamina = "widget_weekly_avg_stamina"
        static let weeklyDayData = "widget_weekly_day_data"
        static let lastUpdated = "widget_last_updated"
    }

    // MARK: - Read

    static func readSnapshot() -> WidgetSnapshot {
        guard let defaults = UserDefaults(suiteName: appGroupID) else { return .empty }

        let lastSessionTime: Date? = {
            let ts = defaults.double(forKey: Key.lastSessionTime)
            return ts > 0 ? Date(timeIntervalSince1970: ts) : nil
        }()

        let weeklyDays: [WidgetSnapshot.DayEntry] = {
            guard let data = defaults.data(forKey: Key.weeklyDayData),
                  let entries = try? JSONDecoder().decode([WidgetSnapshot.DayEntry].self, from: data)
            else { return [] }
            return entries
        }()

        let lastUpdated: Date = {
            let ts = defaults.double(forKey: Key.lastUpdated)
            return ts > 0 ? Date(timeIntervalSince1970: ts) : Date()
        }()

        return WidgetSnapshot(
            todaySessionCount: defaults.integer(forKey: Key.todaySessionCount),
            todayTotalMin: defaults.integer(forKey: Key.todayTotalMin),
            todayAvgStamina: defaults.double(forKey: Key.todayAvgStamina),
            bestSlotName: defaults.string(forKey: Key.bestSlotName),
            bestSlotAvg: defaults.double(forKey: Key.bestSlotAvg),
            lastSessionTime: lastSessionTime,
            lastSessionStamina: defaults.double(forKey: Key.lastSessionStamina),
            weeklyAvgStamina: defaults.double(forKey: Key.weeklyAvgStamina),
            weeklyDays: weeklyDays,
            lastUpdated: lastUpdated
        )
    }
}
