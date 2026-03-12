import Foundation
import QuartzCore
import os.signpost

/// 轻量级性能监测器
/// - 启动耗时（进程创建 → 首帧渲染）
/// - 实时帧率 & 掉帧统计（CADisplayLink）
/// - 内存占用
@MainActor
final class PerformanceMonitor: ObservableObject {

    static let shared = PerformanceMonitor()

    // MARK: - Launch Metrics

    /// 进程创建时间（内核记录）
    let processStartTime: TimeInterval
    /// 标记 App.init() 时间
    private(set) var appInitTime: TimeInterval = 0
    /// 标记首帧渲染时间
    private(set) var firstFrameTime: TimeInterval = 0

    var launchDuration: TimeInterval {
        guard firstFrameTime > 0 else { return 0 }
        return firstFrameTime - processStartTime
    }

    var launchDurationMs: Int { Int(launchDuration * 1000) }

    // MARK: - Runtime Metrics

    @Published private(set) var fps: Int = 0
    @Published private(set) var droppedFrames: Int = 0
    @Published private(set) var memoryMB: Double = 0
    @Published private(set) var isMonitoring = false

    private var displayLink: CADisplayLink?
    private var lastTimestamp: CFTimeInterval = 0
    private var frameCount: Int = 0
    private var fpsUpdateInterval: CFTimeInterval = 1.0 // 每秒更新一次 FPS
    private var totalDropped: Int = 0

    private let logger = Logger(subsystem: "com.fluxchi.app", category: "Performance")

    // MARK: - Init

    private init() {
        self.processStartTime = Self.getProcessStartTime()
    }

    // MARK: - Launch Tracking

    func markAppInit() {
        appInitTime = CACurrentMediaTime()
    }

    func markFirstFrame() {
        guard firstFrameTime == 0 else { return }
        firstFrameTime = CACurrentMediaTime()
        logger.info("🚀 Launch: \(self.launchDurationMs)ms (process→first frame)")
    }

    // MARK: - FPS Monitoring

    func startMonitoring() {
        guard !isMonitoring else { return }
        isMonitoring = true
        droppedFrames = 0
        totalDropped = 0

        let link = CADisplayLink(target: DisplayLinkProxy(monitor: self),
                                 selector: #selector(DisplayLinkProxy.tick(_:)))
        link.add(to: .main, forMode: .common)
        displayLink = link
        lastTimestamp = 0
        frameCount = 0

        // 定期更新内存
        updateMemory()
    }

    func stopMonitoring() {
        displayLink?.invalidate()
        displayLink = nil
        isMonitoring = false
    }

    fileprivate func handleDisplayLink(_ link: CADisplayLink) {
        if lastTimestamp == 0 {
            lastTimestamp = link.timestamp
            frameCount = 0
            return
        }

        frameCount += 1

        // 检测掉帧：如果两帧间隔 > 预期间隔的 1.5 倍
        let expectedInterval = link.targetTimestamp - link.timestamp
        let actualInterval = link.timestamp - lastTimestamp
        if actualInterval > expectedInterval * 1.5 && expectedInterval > 0 {
            let dropped = Int((actualInterval / expectedInterval).rounded()) - 1
            totalDropped += max(dropped, 0)
        }

        let elapsed = link.timestamp - (lastTimestamp == 0 ? link.timestamp : (link.timestamp - actualInterval + fpsUpdateInterval > link.timestamp ? lastTimestamp : lastTimestamp))

        // 每秒更新 FPS
        let timeSinceLastUpdate = link.timestamp - lastTimestamp
        if timeSinceLastUpdate >= fpsUpdateInterval {
            let currentFPS = Int(Double(frameCount) / timeSinceLastUpdate)
            fps = min(currentFPS, Int(1.0 / expectedInterval) + 1) // cap at display max
            droppedFrames = totalDropped
            frameCount = 0
            lastTimestamp = link.timestamp
            updateMemory()
        }
    }

    // MARK: - Memory

    private func updateMemory() {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if result == KERN_SUCCESS {
            memoryMB = Double(info.resident_size) / 1_048_576
        }
    }

    // MARK: - Process Start Time

    private static func getProcessStartTime() -> TimeInterval {
        var kinfo = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.size
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        guard sysctl(&mib, UInt32(mib.count), &kinfo, &size, nil, 0) == 0 else {
            return CACurrentMediaTime()
        }
        let startSec = kinfo.kp_proc.p_starttime.tv_sec
        let startUsec = kinfo.kp_proc.p_starttime.tv_usec
        let processStart = TimeInterval(startSec) + TimeInterval(startUsec) / 1_000_000

        // 转换为 CACurrentMediaTime 基准
        let wallNow = Date().timeIntervalSince1970
        let machNow = CACurrentMediaTime()
        return machNow - (wallNow - processStart)
    }
}

// MARK: - DisplayLink Proxy (避免 retain cycle + @MainActor 限制)

private class DisplayLinkProxy {
    weak var monitor: PerformanceMonitor?

    init(monitor: PerformanceMonitor) {
        self.monitor = monitor
    }

    @objc func tick(_ link: CADisplayLink) {
        Task { @MainActor [weak self] in
            self?.monitor?.handleDisplayLink(link)
        }
    }
}
