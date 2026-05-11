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

    private var lastState: StaminaState = .focused

    func requestPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { _, _ in }
    }

    /// 推荐入口：直接传 `StaminaState` 枚举。
    func evaluate(stamina: Double, state: StaminaState, continuousWorkMin: Double) {
        guard alertsEnabled else { return }

        switch state {
        case .fading:
            if fadingStart == nil { fadingStart = Date() }
            if let start = fadingStart, Date().timeIntervalSince(start) > fadingSustain {
                fire(
                    title: "专注度持续下降",
                    body: "已连续工作 \(Int(continuousWorkMin)) 分钟，Stamina \(Int(stamina))。建议休息 5-10 分钟后再继续。",
                    flash: false
                )
            }

        case .depleted:
            fire(
                title: "需要休息",
                body: "Stamina 仅剩 \(Int(stamina))，继续工作效率会很低。现在休息一下，让身体恢复。",
                flash: true
            )

        case .focused, .recovering:
            fadingStart = nil
        }

        if lastState == .focused && state == .fading {
            haptic(.warning)
        }
        lastState = state
    }

    /// 兼容旧调用点：字符串自动解析；解析不出来按 `.focused` 处理（不触发提醒）。
    func evaluate(stamina: Double, state: String, continuousWorkMin: Double) {
        let parsed = StaminaState(rawValue: state) ?? .focused
        evaluate(stamina: stamina, state: parsed, continuousWorkMin: continuousWorkMin)
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

    /// 闪烁手电筒。
    /// - 注意：调用 `AVCaptureDevice.default(for: .video)` 会触发摄像头权限弹窗。Info.plist 必须声明
    ///   `NSCameraUsageDescription`，否则会被系统直接拒绝、闪光不生效。
    /// - 锁定后 `defer` 兜底解锁，防止 Task cancel/error 时持锁泄漏。
    private func flashTorch() {
        guard let device = AVCaptureDevice.default(for: .video), device.hasTorch else {
            FluxLog.app.debug("flashTorch: 无可用 torch 设备")
            return
        }

        // 摄像头授权：未授权时不要弹一次性的请求（避免在通知场景里强行弹权限），直接放弃。
        let auth = AVCaptureDevice.authorizationStatus(for: .video)
        guard auth == .authorized else {
            FluxLog.app.debug("flashTorch: 摄像头未授权（status=\(auth.rawValue)），跳过闪光")
            return
        }

        Task.detached {
            do {
                try device.lockForConfiguration()
            } catch {
                FluxLog.app.warn("flashTorch: lockForConfiguration 失败 — \(error.localizedDescription)")
                return
            }
            defer { device.unlockForConfiguration() }

            for _ in 0..<3 {
                try? device.setTorchModeOn(level: 1.0)
                try? await Task.sleep(for: .milliseconds(150))
                if Task.isCancelled { break }
                device.torchMode = .off
                try? await Task.sleep(for: .milliseconds(150))
                if Task.isCancelled { break }
            }
        }
    }
}
