import Foundation
import HealthKit
import SwiftData

/**
 HealthKitManager - 将 FocuX 数据同步到 Apple Health

 核心理念：成为"身体数据"的基础设施层
 - 将专注会话写入 HealthKit
 - 与其他健康数据（睡眠、心率、活动）关联
 - 用户可以在 Apple Health 中看到完整健康图景
 */

@available(iOS 17.0, *)
final class HealthKitManager {

    static let shared = HealthKitManager()

    private let healthStore = HKHealthStore()

    // MARK: - HealthKit 数据类型

    /// 我们可以写入的数据类型
    var writableTypes: Set<HKSampleType> {
        var types = Set<HKSampleType>()

        // 专注/正念会话
        if #available(iOS 17.0, *) {
            types.insert(HKObjectType.workoutType())
        }

        // 活动能量（估算肌肉活动消耗）
        types.insert(HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned)!)

        // 心率（如果有同步的数据）
        types.insert(HKQuantityType.quantityType(forIdentifier: .heartRate)!)

        return types
    }

    // MARK: - 请求权限

    func requestAuthorization(completion: @escaping (Bool, Error?) -> Void) {
        guard HKHealthStore.isHealthDataAvailable() else {
            completion(false, nil)
            return
        }

        healthStore.requestAuthorization(toShare: writableTypes, read: nil) { success, error in
            DispatchQueue.main.async {
                completion(success, error)
            }
        }
    }

    // MARK: - 写入会话

    /// 将专注会话同步到 HealthKit
    func syncSession(_ session: Session, completion: @escaping (Bool, Error?) -> Void) {
        guard HKHealthStore.isHealthDataAvailable() else {
            completion(false, nil)
            return
        }

        var samples: [HKSample] = []

        // 1. 创建 Workout（专注会话）
        if let workout = createWorkout(from: session) {
            samples.append(workout)
        }

        // 2. 创建活动能量样本（基于肌肉活动强度）
        if let energySamples = createEnergySamples(from: session) {
            samples.append(contentsOf: energySamples)
        }

        guard !samples.isEmpty else {
            completion(false, nil)
            return
        }

        // 批量保存
        healthStore.save(samples) { success, error in
            DispatchQueue.main.async {
                completion(success, error)
            }
        }
    }

    // MARK: - 创建 Workout

    private func createWorkout(from session: Session) -> HKWorkout? {
        let config = HKWorkoutConfiguration()
        config.activityType = .preparationAndRecovery  // 使用准备/恢复类型（最接近"专注"）
        config.locationType = .indoor

        // 计算总时长
        let duration = session.duration
        guard duration > 0 else { return nil }

        // 计算估算能量消耗（基于平均续航和时长）
        let avgStamina = session.avgStamina ?? 50
        let caloriesPerMinute = Double(avgStamina / 100 * 2)  // 估算：续航100% = 2卡/分钟
        let totalEnergy = duration / 60 * caloriesPerMinute

        let workout = HKWorkout(
            activityType: .preparationAndRecovery,
            start: session.startedAt,
            end: session.endedAt ?? Date(),
            workoutEvents: nil,
            totalEnergyBurned: totalEnergy > 0 ? HKQuantity(unit: HKUnit.kilocalorie(), doubleValue: totalEnergy) : nil,
            totalDistance: nil,
            metadata: [
                "FocuX_Session_ID": session.id.uuidString,
                "FocuX_Avg_Stamina": avgStamina,
                "FocuX_Source": session.source.displayName,
                "FocuX_Title": session.title
            ]
        )

        return workout
    }

    // MARK: - 创建能量样本

    private func createEnergySamples(from session: Session) -> [HKSample]? {
        guard let avgStamina = session.avgStamina else { return nil }

        // 将会话分成 5 分钟一段，每段记录能量消耗
        let interval: TimeInterval = 5 * 60  // 5 分钟
        let segments = Int(session.duration / interval)

        guard segments > 0 else { return nil }

        var samples: [HKSample] = []
        let startDate = session.startedAt

        for i in 0..<segments {
            let startTime = startDate.addingTimeInterval(TimeInterval(i) * interval)
            let endTime = min(startTime.addingTimeInterval(interval), session.endedAt ?? Date())

            // 基于该时间段的活动计算能量
            let caloriesPerMinute = Double(avgStamina / 100 * 2)
            let segmentEnergy = (endTime.timeIntervalSince(startTime) / 60) * caloriesPerMinute

            let quantity = HKQuantity(
                unit: HKUnit.kilocalorie(),
                doubleValue: segmentEnergy
            )

            let sample = HKQuantitySample(
                type: HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned)!,
                quantity: quantity,
                start: startTime,
                end: endTime,
                metadata: [
                    "FocuX_Session_ID": session.id.uuidString,
                    "FocuX_Segment": i
                ]
            )

            samples.append(sample)
        }

        return samples.isEmpty ? nil : samples
    }

    // MARK: - 从 HealthKit 读取关联数据

    /// 读取最近 7 天的睡眠数据（用于关联分析）
    func fetchRecentSleep(completion: @escaping ([HKCategorySample]?) -> Void) {
        guard let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) else {
            completion(nil)
            return
        }

        let now = Date()
        let startOfDay = Calendar.current.startOfDay(for: now)
        let sevenDaysAgo = Calendar.current.date(byAdding: .day, value: -7, to: startOfDay)!

        let predicate = HKQuery.predicateForSamples(withStart: sevenDaysAgo, end: now, options: .strictStartDate)

        let query = HKSampleQuery(
            sampleType: sleepType,
            predicate: predicate,
            limit: HKObjectQueryNoLimit,
            sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]
        ) { _, samples, _ in
            DispatchQueue.main.async {
                completion(samples as? [HKCategorySample])
            }
        }

        healthStore.execute(query)
    }

    /// 读取最近的活动数据（用于对比分析）
    func fetchRecentWorkouts(completion: @escaping ([HKWorkout]?) -> Void) {
        let now = Date()
        let sevenDaysAgo = Calendar.current.date(byAdding: .day, value: -7, to: now)!

        let predicate = HKQuery.predicateForSamples(withStart: sevenDaysAgo, end: now, options: .strictStartDate)

        let query = HKSampleQuery(
            sampleType: HKWorkoutType.workoutType(),
            predicate: predicate,
            limit: HKObjectQueryNoLimit,
            sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]
        ) { _, samples, _ in
            DispatchQueue.main.async {
                completion(samples as? [HKWorkout])
            }
        }

        healthStore.execute(query)
    }

    // MARK: - 导出 HealthKit 数据

    /// 导出 JSON 格式（用于备份或分析）
    func exportHealthData(completion: @escaping (URL?) -> Void) {
        // 读取最近 30 天的所有健康数据
        let now = Date()
        let thirtyDaysAgo = Calendar.current.date(byAdding: .day, value: -30, to: now)!

        var allData: [[String: Any]] = []

        let dispatchGroup = DispatchGroup()

        // 1. 读取 Workout
        dispatchGroup.enter()
        fetchRecentWorkouts { workouts in
            defer { dispatchGroup.leave() }
            guard let workouts = workouts else { return }

            for workout in workouts {
                allData.append([
                    "type": "workout",
                    "startDate": workout.startDate.iso8601String,
                    "endDate": (workout.endDate).iso8601String,
                    "duration": workout.duration,
                    "energyBurned": workout.totalEnergyBurned?.doubleValue(for: HKUnit.kilocalorie()),
                    "activityType": workout.workoutActivityType.rawValue,
                    "metadata": workout.metadata ?? [:]
                ])
            }
        }

        // 2. 读取睡眠
        dispatchGroup.enter()
        fetchRecentSleep { sleeps in
            defer { dispatchGroup.leave() }
            guard let sleeps = sleeps else { return }

            for sleep in sleeps {
                allData.append([
                    "type": "sleep",
                    "startDate": sleep.startDate.iso8601String,
                    "endDate": sleep.endDate.iso8601String,
                    "duration": sleep.endDate.timeIntervalSince(sleep.startDate),
                    "value": sleep.value
                ])
            }
        }

        dispatchGroup.notify(queue: .main) {
            guard !allData.isEmpty else {
                completion(nil)
                return
            }

            do {
                let data = try JSONSerialization.data(withJSONObject: allData, options: .prettyPrinted)
                let url = FileManager.default.temporaryDirectory.appendingPathComponent("focux_health_export_\(Date().timeIntervalSince1970).json")
                try data.write(to: url)
                completion(url)
            } catch {
                completion(nil)
            }
        }
    }
}

// MARK: - Date Extension

extension Date {
    var iso8601String: String {
        let formatter = ISO8601DateFormatter()
        return formatter.string(from: self)
    }
}
