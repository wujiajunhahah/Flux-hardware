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
}

// MARK: - Live Activity Manager

@MainActor
final class LiveActivityManager: ObservableObject {

    @Published var isActive = false
    private var activity: Activity<FluxChiLiveAttributes>?
    private let log = Logger(subsystem: "com.fluxchi", category: "LiveActivity")

    func startActivity(title: String) {
        guard ActivityAuthorizationInfo().areActivitiesEnabled else {
            log.info("LiveActivity 未授权")
            return
        }

        let attributes = FluxChiLiveAttributes(sessionTitle: title)

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
            log.info("LiveActivity 已启动: \(self.activity?.id ?? "nil")")
        } catch {
            log.error("LiveActivity 启动失败: \(error.localizedDescription)")
        }
    }

    func updateActivity(stamina: Double, state: String,
                        activity act: String, consistency: Double,
                        tension: Double, fatigue: Double) {
        guard let activity else { return }

        let updatedState = FluxChiLiveAttributes.ContentState(
            stamina: stamina,
            state: state,
            activity: act,
            consistency: consistency,
            tension: tension,
            fatigue: fatigue
        )

        let content = ActivityContent(state: updatedState, staleDate: Date.now.addingTimeInterval(30))

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
