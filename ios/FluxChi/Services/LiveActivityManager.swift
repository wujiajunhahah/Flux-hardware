import Foundation
import ActivityKit
import SwiftUI

// MARK: - Live Activity Attributes

struct FluxChiLiveAttributes: ActivityAttributes {
    public struct ContentState: Codable, Hashable {
        var stamina: Double
        var state: String
        var continuousWorkMin: Double
        var activity: String
        var consistency: Double
        var tension: Double
        var fatigue: Double
    }

    var sessionTitle: String
    var startedAt: Date
}

// MARK: - Live Activity Manager

@MainActor
final class LiveActivityManager: ObservableObject {

    @Published var isActive = false
    private var activity: Activity<FluxChiLiveAttributes>?

    func startActivity(title: String) {
        guard ActivityAuthorizationInfo().areActivitiesEnabled else {
            print("[LiveActivity] Not authorized")
            return
        }

        let attributes = FluxChiLiveAttributes(
            sessionTitle: title,
            startedAt: Date()
        )

        let initialState = FluxChiLiveAttributes.ContentState(
            stamina: 100,
            state: "focused",
            continuousWorkMin: 0,
            activity: "rest",
            consistency: 0,
            tension: 0,
            fatigue: 0
        )

        let content = ActivityContent(state: initialState, staleDate: nil)

        do {
            activity = try Activity.request(
                attributes: attributes,
                content: content,
                pushType: nil
            )
            isActive = true
            print("[LiveActivity] Started: \(activity?.id ?? "nil")")
        } catch {
            print("[LiveActivity] Failed to start: \(error)")
        }
    }

    func updateActivity(stamina: Double, state: String, workMin: Double,
                        activity act: String, consistency: Double,
                        tension: Double, fatigue: Double) {
        guard let activity else { return }

        let updatedState = FluxChiLiveAttributes.ContentState(
            stamina: stamina,
            state: state,
            continuousWorkMin: workMin,
            activity: act,
            consistency: consistency,
            tension: tension,
            fatigue: fatigue
        )

        let content = ActivityContent(state: updatedState, staleDate: nil)

        Task {
            await activity.update(content)
        }
    }

    func endActivity() {
        guard let activity else { return }

        let finalState = FluxChiLiveAttributes.ContentState(
            stamina: 0,
            state: "ended",
            continuousWorkMin: 0,
            activity: "rest",
            consistency: 0,
            tension: 0,
            fatigue: 0
        )

        let content = ActivityContent(state: finalState, staleDate: nil)

        Task {
            await activity.end(content, dismissalPolicy: .default)
        }

        self.activity = nil
        isActive = false
    }
}
