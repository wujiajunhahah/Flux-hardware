import Foundation
import UserNotifications
import AVFoundation
import UIKit

@MainActor
final class AlertManager: ObservableObject {

    @Published var showBreakAlert = false
    @Published var alertTitle = ""
    @Published var alertMessage = ""
    @Published var alertsEnabled = true

    private var lastAlertTime: Date?
    private let minInterval: TimeInterval = 600 // 10 min cooldown

    private var fadingStart: Date?
    private let fadingSustain: TimeInterval = 120 // 2 min sustained fading → alert

    private var lastState = "focused"

    func requestPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { _, _ in }
    }

    func evaluate(stamina: Double, state: String, continuousWorkMin: Double) {
        guard alertsEnabled else { return }

        switch state {
        case "fading":
            if fadingStart == nil { fadingStart = Date() }
            if let start = fadingStart, Date().timeIntervalSince(start) > fadingSustain {
                fire(
                    title: "专注度持续下降",
                    body: "已连续工作 \(Int(continuousWorkMin)) 分钟，Stamina \(Int(stamina))。建议休息 5-10 分钟后再继续。",
                    flash: false
                )
            }

        case "depleted":
            fire(
                title: "需要休息",
                body: "Stamina 仅剩 \(Int(stamina))，继续工作效率会很低。现在休息一下，让身体恢复。",
                flash: true
            )

        default:
            fadingStart = nil
        }

        if lastState == "focused" && state == "fading" {
            haptic(.warning)
        }
        lastState = state
    }

    func dismissAlert() {
        showBreakAlert = false
    }

    // MARK: - Private

    private func fire(title: String, body: String, flash: Bool) {
        if let last = lastAlertTime, Date().timeIntervalSince(last) < minInterval { return }
        lastAlertTime = Date()

        alertTitle = title
        alertMessage = body
        showBreakAlert = true

        sendNotification(title: title, body: body)
        haptic(.error)
        if flash { flashTorch() }
    }

    private func sendNotification(title: String, body: String) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default
        content.userInfo = ["action": "showActiveSession", "showRest": true]
        content.categoryIdentifier = "FOCUS_ALERT"
        let req = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(req)
    }

    private func haptic(_ type: UINotificationFeedbackGenerator.FeedbackType) {
        UINotificationFeedbackGenerator().notificationOccurred(type)
    }

    private func flashTorch() {
        guard let device = AVCaptureDevice.default(for: .video), device.hasTorch else { return }
        Task.detached {
            try? device.lockForConfiguration()
            for _ in 0..<3 {
                try? device.setTorchModeOn(level: 1.0)
                try? await Task.sleep(for: .milliseconds(150))
                device.torchMode = .off
                try? await Task.sleep(for: .milliseconds(150))
            }
            device.unlockForConfiguration()
        }
    }
}
