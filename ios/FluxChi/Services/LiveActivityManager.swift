import Foundation
import ActivityKit
import os.log
import SwiftUI

// MARK: - Live Activity Attributes

struct FluxChiLiveAttributes: ActivityAttributes {
    public struct ContentState: Codable, Hashable {
        var stamina: Double
        var state: String
        var activity: String
        var consistency: Double
        var tension: Double
        var fatigue: Double
    }

    var sessionTitle: String
    /// 会话开始时刻；通过 `Text(_:style:.timer)` 自动跳秒显示，无需 LiveActivityManager 推 state
    /// 仅为更新计时器。
    var startedAt: Date
}

// MARK: - Live Activity Manager

@MainActor
final class LiveActivityManager: ObservableObject {

    @Published var isActive = false
    private var activity: Activity<FluxChiLiveAttributes>?
    private let log = Logger(subsystem: "com.fluxchi", category: "LiveActivity")

    /// 节流到 ≤1Hz：Apple 文档建议 LiveActivity 更新频率不超过 1Hz，否则系统会节流推送、
    /// 主进程到 ActivityKit 守护进程的 XPC 也会拥塞导致 MainActor 卡顿。
    /// 上游数据是 5Hz（BLE 直连）或 1-2Hz（REST 轮询），按时间窗合并到 1Hz。
    private var lastUpdateAt: Date?
    private let minUpdateInterval: TimeInterval = 1.0

    func startActivity(title: String) {
        guard ActivityAuthorizationInfo().areActivitiesEnabled else {
            log.info("LiveActivity 未授权")
            return
        }

        let attributes = FluxChiLiveAttributes(sessionTitle: title, startedAt: Date())

        let initialState = FluxChiLiveAttributes.ContentState(
            stamina: 100,
            state: "focused",
            activity: "rest",
            consistency: 0,
            tension: 0,
            fatigue: 0
        )

        let content = ActivityContent(
            state: initialState,
            staleDate: Date.now.addingTimeInterval(60)
        )

        do {
            activity = try Activity.request(
                attributes: attributes,
                content: content,
                pushType: nil
            )
            // 设置 widgetURL 以便点击灵动岛跳转到 ActiveSessionView
            // 注：widgetURL 需要在 Widget Extension 的 View 中设置
            // 这里通过 URL Scheme 支持 App 端处理
            isActive = true
            lastUpdateAt = nil  // 重置节流窗口，首帧立即推送
            log.info("LiveActivity 已启动: \(self.activity?.id ?? "nil")")
        } catch {
            log.error("LiveActivity 启动失败: \(error.localizedDescription)")
        }
    }

    func updateActivity(stamina: Double, state: String,
                        activity act: String, consistency: Double,
                        tension: Double, fatigue: Double) {
        guard let activity else { return }

        // 1Hz 节流：丢弃距上次推送 < minUpdateInterval 的更新。
        // 状态变更敏感度不高于 1 秒，UI 视感无差异，但能消除 5Hz × XPC 带来的主线程压力。
        let now = Date()
        if let last = lastUpdateAt, now.timeIntervalSince(last) < minUpdateInterval {
            return
        }
        lastUpdateAt = now

        let updatedState = FluxChiLiveAttributes.ContentState(
            stamina: stamina,
            state: state,
            activity: act,
            consistency: consistency,
            tension: tension,
            fatigue: fatigue
        )

        let content = ActivityContent(state: updatedState, staleDate: now.addingTimeInterval(30))

        Task {
            await activity.update(content)
        }
    }

    func endActivity() {
        guard let activity else { return }

        let finalState = FluxChiLiveAttributes.ContentState(
            stamina: 0,
            state: "ended",
            activity: "rest",
            consistency: 0,
            tension: 0,
            fatigue: 0
        )

        let content = ActivityContent(state: finalState, staleDate: nil)

        Task {
            await activity.end(content, dismissalPolicy: .immediate)
        }

        self.activity = nil
        isActive = false
    }
}
