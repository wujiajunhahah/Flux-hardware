import Foundation
import WidgetKit

/// 主 App 与 Widget Extension 之间的共享数据层
/// 通过 App Group UserDefaults 传递摘要数据，避免 SwiftData 跨进程共享的复杂性
enum WidgetDataManager {

    static let appGroupID = "group.com.fluxchi.app"

    private static var sharedDefaults: UserDefaults? {
        UserDefaults(suiteName: appGroupID)
    }

    // MARK: - Keys

    private enum Key {
        static let todaySessionCount = "widget_today_session_count"
        static let todayTotalMin = "widget_today_total_min"
        static let todayAvgStamina = "widget_today_avg_stamina"
        static let bestSlotName = "widget_best_slot_name"
        static let bestSlotAvg = "widget_best_slot_avg"
        static let lastSessionTime = "widget_last_session_time"
        static let lastSessionStamina = "widget_last_session_stamina"
        static let weeklyAvgStamina = "widget_weekly_avg_stamina"
        static let weeklyDayData = "widget_weekly_day_data" // JSON: [{date, avg}]
        static let lastUpdated = "widget_last_updated"
    }

    // MARK: - Data Model (Widget 读取用)

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

        /// 空快照（无数据时使用）
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

    // MARK: - Write (主 App 调用)

    static func updateFromSessions(today: [some SessionDataProvider], recent: [some SessionDataProvider]) {
        guard let defaults = sharedDefaults else { return }

        let todayCount = today.count
        let todayTotalMin = Int(today.reduce(0) { $0 + $1.sessionDuration } / 60)
        let todayStaminas = today.compactMap(\.sessionAvgStamina)
        let todayAvg = todayStaminas.isEmpty ? 0.0 : todayStaminas.reduce(0, +) / Double(todayStaminas.count)

        defaults.set(todayCount, forKey: Key.todaySessionCount)
        defaults.set(todayTotalMin, forKey: Key.todayTotalMin)
        defaults.set(todayAvg, forKey: Key.todayAvgStamina)

        // 最佳时段（先按 slot 分桶 → 求平均 → 一次 max；避免 O(n²)）
        var slotSums: [Flux.TimeSlot: (sum: Double, n: Int)] = [:]
        for s in today {
            guard let avg = s.sessionAvgStamina else { continue }
            let slot = Flux.TimeSlot.from(date: s.sessionStartedAt)
            let cur = slotSums[slot] ?? (0, 0)
            slotSums[slot] = (cur.sum + avg, cur.n + 1)
        }
        let slotAvgs: [(slot: Flux.TimeSlot, avg: Double)] = slotSums.map { ($0.key, $0.value.sum / Double($0.value.n)) }
        if let best = slotAvgs.max(by: { $0.avg < $1.avg }) {
            defaults.set(best.slot.rawValue, forKey: Key.bestSlotName)
            defaults.set(best.avg, forKey: Key.bestSlotAvg)
        } else {
            defaults.removeObject(forKey: Key.bestSlotName)
            defaults.removeObject(forKey: Key.bestSlotAvg)
        }

        // 最近一次
        if let last = today.sorted(by: { $0.sessionStartedAt < $1.sessionStartedAt }).last {
            defaults.set(last.sessionStartedAt.timeIntervalSince1970, forKey: Key.lastSessionTime)
            defaults.set(last.sessionAvgStamina ?? 0, forKey: Key.lastSessionStamina)
        } else {
            defaults.removeObject(forKey: Key.lastSessionTime)
            defaults.removeObject(forKey: Key.lastSessionStamina)
        }

        // 周数据
        let recentStaminas = recent.compactMap(\.sessionAvgStamina)
        let weeklyAvg = recentStaminas.isEmpty ? 0.0 : recentStaminas.reduce(0, +) / Double(recentStaminas.count)
        defaults.set(weeklyAvg, forKey: Key.weeklyAvgStamina)

        // 每日聚合
        let cal = Calendar.current
        let grouped = Dictionary(grouping: recent) { cal.startOfDay(for: $0.sessionStartedAt) }
        let dayEntries: [WidgetSnapshot.DayEntry] = grouped.map { (date, sessions) in
            let avgs = sessions.compactMap(\.sessionAvgStamina)
            let avg = avgs.isEmpty ? 0.0 : avgs.reduce(0, +) / Double(avgs.count)
            return WidgetSnapshot.DayEntry(date: date, avgStamina: avg, sessionCount: sessions.count)
        }.sorted { $0.date < $1.date }

        if let data = try? JSONEncoder().encode(dayEntries) {
            defaults.set(data, forKey: Key.weeklyDayData)
        }

        defaults.set(Date().timeIntervalSince1970, forKey: Key.lastUpdated)

        // 通知 WidgetKit 刷新
        WidgetCenter.shared.reloadAllTimelines()
    }

    // MARK: - Read (Widget 调用)

    static func readSnapshot() -> WidgetSnapshot {
        guard let defaults = sharedDefaults else { return .empty }

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

// MARK: - Protocol (避免 Widget Extension 依赖 SwiftData Session)

protocol SessionDataProvider {
    var sessionStartedAt: Date { get }
    var sessionDuration: TimeInterval { get }
    var sessionAvgStamina: Double? { get }
}

// 让 Session 遵循协议
extension Session: SessionDataProvider {
    var sessionStartedAt: Date { startedAt }
    var sessionDuration: TimeInterval { duration }
    var sessionAvgStamina: Double? { avgStamina }
}
