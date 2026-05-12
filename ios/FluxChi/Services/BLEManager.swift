import Foundation
import CoreBluetooth
import Combine
import OSLog

// BLE constants extracted to a nonisolated enum so CBCentralManager / CBPeripheral
// delegate callbacks (which run nonisolated) can reference them without crossing
// the @MainActor boundary.  Fixes Swift 6 strict-concurrency diagnostics.
private enum BLEConstants {
    static let serviceUUID   = CBUUID(string: "974CBE30-3E83-465E-ACDE-6F92FE712134")
    static let dataCharUUID  = CBUUID(string: "974CBE31-3E83-465E-ACDE-6F92FE712134")
    static let writeCharUUID = CBUUID(string: "974CBE32-3E83-465E-ACDE-6F92FE712134")
    static let batteryServiceUUID = CBUUID(string: "180F")
    static let batteryCharUUID    = CBUUID(string: "2A19")
    static let devicePrefix  = "WL"
    static let emgFlag: UInt8 = 0xAA
    static let imuFlag: UInt8 = 0xBB
    /// 与固件一致：最多 8 路 24-bit EMG（µV）；旧版 6 路仍兼容
    static let channelCount  = 8
    /// 脱落检测：µV 尺度下全通道 RMS 极低（与 Python 流直读 µV 对齐）
    static let detachRMSThreshold: Double = 25.0
    /// 脱落检测：需持续低信号的帧数（约 30 秒，50帧/s × 30s ÷ 每50帧检测一次 = 30次）
    static let detachConsecutiveCount = 30
    /// CoreBluetooth state restoration 标识；与 Info.plist 的 `bluetooth-central` background mode 配套，
    /// 让系统在 app 被挂起后唤回时恢复正在监听的 peripheral。
    static let centralRestoreIdentifier = "com.fluxchi.ble.central"
}

// MARK: - Notification Names

extension Notification.Name {
    static let bleDeviceDetached = Notification.Name("BLEManager.deviceDetached")
    static let fluxShowRestFromNotification = Notification.Name("FluxChi.showRestFromNotification")
}

@MainActor
final class BLEManager: NSObject, ObservableObject {

    // MARK: - Published

    @Published var peripheralState: CBPeripheralState = .disconnected
    @Published var discoveredDevices: [CBPeripheral] = []
    @Published var deviceRSSI: [UUID: Int] = [:]
    @Published var connectedDeviceName: String?
    @Published var isScanning = false
    @Published var latestRMS: [Double] = Array(repeating: 0, count: 8)
    @Published var emgFrameCount: Int = 0
    @Published var activeChannelCount: Int = 8
    @Published var batteryLevel: Int? // 0-100，nil = 未知
    /// 实测 BLE 帧率（rolling 64 帧）。WAVELETECH BLE 标称 1000Hz，实际受蓝牙拥塞约 100-340Hz。
    /// 没有这个测量、用硬编码 320Hz 算 FFT，会让 MNF/MDF 频率轴整体偏移 ×3 倍以上。
    @Published private(set) var measuredSampleRateHz: Double = 320

    /// IMU 运动幅度 — 滚动均值的 ||gyro|| (rad/s)。
    /// 用途：辅助 work/rest 判定（手在动 ≠ 安静）+ 抑制运动伪迹（高 IMU 期间 EMG fatigue 不可信）。
    /// 之前 IMU 帧（0xBB）被完全丢弃，这是巨大的浪费——硬件已经发了，我们没用上。
    @Published private(set) var imuMotion: Double = 0

    var isConnected: Bool { peripheralState == .connected }

    // MARK: - Callbacks

    var onEMGSample: (([Double]) -> Void)?
    var onStateUpdate: ((FluxState) -> Void)?

    // MARK: - Private

    private var central: CBCentralManager!
    private var peripheral: CBPeripheral?
    private var dataChar: CBCharacteristic?

    private let emgBuffer = EMGRingBuffer(capacity: 250)
    private let staminaEngine = OnDeviceStaminaEngine()
    private let classifier = EMGActivityInference()
    private var lowSignalConsecutive = 0
    private var detachNotified = false
    /// 标记 ML/FFT compute 是否正在进行；若是，本轮 buildAndPushState 跳过避免 Task 堆积。
    /// 5Hz 触发频率下偶尔丢一帧对 UI 视感无影响，但能防止慢推理把 Task queue 撑爆。
    private var computeInFlight = false

    /// 滚动 64 帧时间戳，用于实时估算 BLE 帧率。
    /// 选 64 而非更大：BLE 帧率本身有抖动，~200ms 窗口在响应快慢之间折中。
    private var frameTimestamps: [TimeInterval] = []
    private let fpsWindowSize = 64
    /// 区分用户主动 disconnect 与意外断连。主动断时不需要 didDisconnect 再清一遍状态。
    private var intentionalDisconnect = false

    // MARK: - Init

    override init() {
        super.init()
        // 用 restore identifier 时，CoreBluetooth 在 init 时就要校验 delegate 是否实现
        // `centralManager(_:willRestoreState:)`。delegate=nil + 之后赋值的两步写法会抛
        // NSException："provided a restore identifier but the delegate doesn't implement..."。
        // 必须 init 时直接把 self 传进去。
        central = CBCentralManager(
            delegate: self,
            queue: nil,
            options: [CBCentralManagerOptionRestoreIdentifierKey: BLEConstants.centralRestoreIdentifier]
        )
    }

    // MARK: - Scanning

    func startScan() {
        guard central.state == .poweredOn else {
            FluxLog.ble.warn("启动扫描失败: 蓝牙未就绪 (state: \(central.state.rawValue))")
            return
        }
        discoveredDevices.removeAll()
        isScanning = true
        central.scanForPeripherals(
            withServices: nil,
            options: [CBCentralManagerScanOptionAllowDuplicatesKey: false]
        )
        FluxLog.ble.info("启动扫描，15秒超时")
        Task {
            try? await Task.sleep(for: .seconds(15))
            if isScanning { stopScan() }
        }
    }

    func stopScan() {
        central.stopScan()
        isScanning = false
        FluxLog.ble.info("停止扫描，发现 \(discoveredDevices.count) 个设备")
    }

    // MARK: - Connect / Disconnect

    func connect(to peripheral: CBPeripheral) {
        stopScan()
        self.peripheral = peripheral
        let deviceName = peripheral.name ?? "未知设备"
        FluxLog.ble.info("发起连接: \(deviceName) [UUID: \(peripheral.identifier.uuidString.prefix(8))]")
        central.connect(peripheral, options: nil)
    }

    func disconnect() {
        guard let p = peripheral else { return }
        intentionalDisconnect = true
        if let c = dataChar { p.setNotifyValue(false, for: c) }
        let deviceName = p.name ?? connectedDeviceName ?? "未知设备"
        FluxLog.ble.info("主动断开: \(deviceName)")
        // 实际状态清理交给 didDisconnect，避免重复 reset 与日志
        central.cancelPeripheralConnection(p)
    }

    // MARK: - EMG Parsing

    private func handleNotification(_ data: Data) {
        guard data.count >= 20 else { return }

        // IMU 帧（0xBB）单独走 IMU 处理路径，不参与 EMG 计算。
        if data[0] == BLEConstants.imuFlag {
            handleIMUFrame(data)
            return
        }
        guard data[0] == BLEConstants.emgFlag else { return }

        let payload = data.dropFirst(2)
        let nCh = min(payload.count / 3, BLEConstants.channelCount)
        activeChannelCount = nCh

        var values = [Double](repeating: 0, count: nCh)
        for i in 0..<nCh {
            let off = i * 3
            let b1 = Int(payload[payload.startIndex + off])
            let b2 = Int(payload[payload.startIndex + off + 1])
            let b3 = Int(payload[payload.startIndex + off + 2])
            values[i] = Self.decodeSigned24(b1, b2, b3)
        }

        emgBuffer.append(values)
        emgFrameCount += 1
        onEMGSample?(values)

        // 滚动 fps 估计：保留最近 64 帧的时间戳，rate = (N-1) / (t_last - t_first)
        let now = Date().timeIntervalSince1970
        frameTimestamps.append(now)
        if frameTimestamps.count > fpsWindowSize { frameTimestamps.removeFirst() }
        if frameTimestamps.count >= 16,
           let first = frameTimestamps.first, let last = frameTimestamps.last,
           last > first {
            let estimated = Double(frameTimestamps.count - 1) / (last - first)
            // 限制在合理范围 [50, 1500] Hz，避免单点抖动把估计带飞
            measuredSampleRateHz = min(1500, max(50, estimated))
        }

        if emgFrameCount % 8 == 0 {
            latestRMS = emgBuffer.rms(window: 24)
        }

        if emgFrameCount % 50 == 0 {
            checkDetachment()
            buildAndPushState()

            let rmsPreview = latestRMS.prefix(3).map { String(format: "%.1f", $0) }.joined(separator: ", ")
            FluxLog.ble.debug("Frame \(emgFrameCount) | RMS: [\(rmsPreview), ...] | Ch: \(nCh) | fps: \(String(format: "%.1f", measuredSampleRateHz))")
        }
    }

    private func buildAndPushState() {
        // Drop-if-in-flight：5Hz 触发 vs ~20-50ms ML+FFT compute，慢推理时跳过本轮避免 Task 堆积。
        guard !computeInFlight else { return }

        // 在 MainActor 上捕获所有输入（emgBuffer / emgFrameCount 都是 MainActor 隔离的）
        let rms = emgBuffer.rms()
        let now = Date().timeIntervalSince1970
        let rawChannels = emgBuffer.channelTimeSeries()
        let frameCount = emgFrameCount
        let sampleRate = measuredSampleRateHz
        let imu = imuMotion  // IMU 运动量，用来辅助 work/rest 判定 + 抑制 MDF 运动伪迹

        computeInFlight = true
        Task { [weak self, staminaEngine, classifier] in
            // 这两个 await 自动从 MainActor 跳到各自的 actor 隔离域，CoreML/FFT 在后台执行
            // 传 measuredSampleRateHz 让 FFT 频率轴与实际 BLE 帧率对齐
            let prediction = await classifier.predict(channels: rawChannels, sampleRateHz: sampleRate)
            let r = await staminaEngine.update(
                rms: rms,
                rawChannels: rawChannels,
                timestamp: now,
                classifiedActivity: prediction?.label,
                sampleRateHz: sampleRate,
                imuMotion: imu
            )

            await MainActor.run {
                guard let self else { return }
                self.computeInFlight = false

                let classifiedActivity = prediction?.label
                let activity = classifiedActivity ?? (r.isWorking ? "working" : "rest")
                let confidence = prediction?.confidence ?? 1.0
                let probs = prediction?.probabilities ?? [activity: 1.0]

                let state = FluxState(
                    timestamp: now,
                    activity: activity,
                    confidence: confidence,
                    probabilities: probs,
                    rms: rms,
                    emgSampleCount: frameCount,
                    stamina: StaminaData(
                        value: r.stamina,
                        state: r.state,
                        consistency: r.consistency,
                        tension: r.tension,
                        fatigue: r.fatigue,
                        drainRate: r.drainRate,
                        recoveryRate: r.recoveryRate,
                        suggestedWorkMin: r.suggestedWorkMin,
                        suggestedBreakMin: r.suggestedBreakMin,
                        continuousWorkMin: r.continuousWorkMin,
                        totalWorkMin: r.totalWorkMin
                    ),
                    decision: DecisionData(
                        state: r.state,
                        recommendation: r.stamina > 60 ? "keep_working" : r.stamina > 30 ? "take_break" : "rest_more",
                        urgency: r.stamina < 30 ? 0.8 : r.stamina < 60 ? 0.5 : 0,
                        reasons: [r.isWorking
                            ? "BLE 直连 · 已工作 \(Int(r.continuousWorkMin)) 分钟 · Stamina \(Int(r.stamina))"
                            : "BLE 直连 · 恢复中 · Stamina \(Int(r.stamina))"],
                        stamina: r.stamina,
                        continuousWorkMin: r.continuousWorkMin,
                        totalWorkMin: r.totalWorkMin,
                        suggestedWorkMin: r.suggestedWorkMin,
                        suggestedBreakMin: r.suggestedBreakMin
                    ),
                    fusion: nil,
                    vision: nil
                )
                self.onStateUpdate?(state)
            }
        }
    }

    // MARK: - Detach Detection

    private func checkDetachment() {
        let maxRMS = latestRMS.max() ?? 0
        if maxRMS < BLEConstants.detachRMSThreshold {
            lowSignalConsecutive += 1
            if lowSignalConsecutive >= BLEConstants.detachConsecutiveCount && !detachNotified {
                detachNotified = true
                NotificationCenter.default.post(name: .bleDeviceDetached, object: nil)
            }
        } else {
            lowSignalConsecutive = 0
            detachNotified = false
        }
    }

    // MARK: - IMU Parsing

    /// IMU 帧：`[0xBB][seq][gyro x/y/z BE int16][accel x/y/z BE int16][pad]`
    /// 缩放系数与 `src/stream.py` 对齐：gyro × 0.0012 → rad/s，accel × 0.0005978 → m/s²
    private func handleIMUFrame(_ data: Data) {
        guard data.count >= 14 else { return }
        let start = data.startIndex + 2  // 跳过 flag + seq

        let g0 = Self.decodeBigEndianInt16(data, start)
        let g1 = Self.decodeBigEndianInt16(data, start + 2)
        let g2 = Self.decodeBigEndianInt16(data, start + 4)

        // 用陀螺仪幅度做运动量，加速度受重力主导噪声大
        // ||gyro|| in rad/s
        let gx = Double(g0) * 0.0012
        let gy = Double(g1) * 0.0012
        let gz = Double(g2) * 0.0012
        let magnitude = (gx * gx + gy * gy + gz * gz).squareRoot()

        // 滚动 EMA 平滑（IMU 噪声大，单帧不可信）
        imuMotion = imuMotion * 0.85 + magnitude * 0.15
    }

    private static func decodeBigEndianInt16(_ data: Data, _ offset: Int) -> Int16 {
        let hi = UInt16(data[offset])
        let lo = UInt16(data[offset + 1])
        return Int16(bitPattern: (hi << 8) | lo)
    }

    /// WAVELETECH：24-bit 大端有符号整数，**值即为 µV**（与 `src/stream.py` 一致，不再做 V_ref/Gain 换算）
    private static func decodeSigned24(_ b1: Int, _ b2: Int, _ b3: Int) -> Double {
        var value = (b1 << 16) | (b2 << 8) | b3
        if value & 0x800000 != 0 { value -= 1 << 24 }
        return Double(value)
    }
}

// MARK: - CBCentralManagerDelegate

extension BLEManager: CBCentralManagerDelegate {

    nonisolated func centralManagerDidUpdateState(_ central: CBCentralManager) {
        Task { @MainActor in _ = central.state }
    }

    /// 系统 restoration：app 在后台被系统终止后再恢复时，CoreBluetooth 会通过此回调把之前的 peripheral 还回来。
    /// 此处只接管 peripheral 引用，让 didConnect/didUpdateValueFor 自然继续。
    nonisolated func centralManager(_ central: CBCentralManager, willRestoreState dict: [String: Any]) {
        guard let restored = dict[CBCentralManagerRestoredStatePeripheralsKey] as? [CBPeripheral],
              let first = restored.first else { return }
        Task { @MainActor in
            FluxLog.ble.info("BLE 状态恢复：接管 \(first.name ?? "未知设备")")
            self.peripheral = first
            first.delegate = self
            self.connectedDeviceName = first.name ?? "WAVELETECH"
            self.peripheralState = first.state
            // 若恢复时仍处 connected，主动重发现服务以补全 dataChar
            if first.state == .connected {
                first.discoverServices([BLEConstants.serviceUUID, BLEConstants.batteryServiceUUID])
            }
        }
    }

    nonisolated func centralManager(
        _ central: CBCentralManager,
        didDiscover peripheral: CBPeripheral,
        advertisementData: [String: Any],
        rssi RSSI: NSNumber
    ) {
        let name = peripheral.name
            ?? (advertisementData[CBAdvertisementDataLocalNameKey] as? String)
            ?? ""
        guard name.uppercased().hasPrefix(BLEConstants.devicePrefix) else { return }
        Task { @MainActor in
            self.deviceRSSI[peripheral.identifier] = RSSI.intValue
            if !self.discoveredDevices.contains(where: { $0.identifier == peripheral.identifier }) {
                self.discoveredDevices.append(peripheral)
                FluxLog.ble.info("发现设备: \(name) [RSSI: \(RSSI)]")
            }
        }
    }

    nonisolated func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        Task { @MainActor in
            self.peripheralState = .connected
            self.connectedDeviceName = peripheral.name ?? "WAVELETECH"
            FluxLog.ble.info("连接成功: \(self.connectedDeviceName ?? "未知设备")")
            peripheral.delegate = self
            peripheral.discoverServices([BLEConstants.serviceUUID, BLEConstants.batteryServiceUUID])
        }
    }

    nonisolated func centralManager(
        _ central: CBCentralManager,
        didDisconnectPeripheral peripheral: CBPeripheral,
        error: Error?
    ) {
        Task { @MainActor in
            let deviceName = self.connectedDeviceName ?? peripheral.name ?? "未知设备"
            let wasIntentional = self.intentionalDisconnect
            self.intentionalDisconnect = false

            self.peripheral = nil
            self.dataChar = nil
            self.connectedDeviceName = nil
            self.peripheralState = .disconnected
            self.emgBuffer.reset()
            self.emgFrameCount = 0
            self.lowSignalConsecutive = 0
            self.detachNotified = false
            self.latestRMS = Array(repeating: 0, count: 8)
            // staminaEngine 是 actor，reset 需要 await
            Task { [staminaEngine = self.staminaEngine] in
                await staminaEngine.reset()
            }

            if wasIntentional {
                FluxLog.ble.info("已断开: \(deviceName)")
            } else if let error = error {
                FluxLog.ble.error("意外断开: \(deviceName) — 状态已完整重置", error: error)
            } else {
                FluxLog.ble.info("断开（系统）: \(deviceName)")
            }
        }
    }
}

// MARK: - CBPeripheralDelegate

extension BLEManager: CBPeripheralDelegate {

    nonisolated func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let services = peripheral.services else { return }
        for service in services {
            if service.uuid == BLEConstants.serviceUUID {
                peripheral.discoverCharacteristics([BLEConstants.dataCharUUID, BLEConstants.writeCharUUID], for: service)
            } else if service.uuid == BLEConstants.batteryServiceUUID {
                peripheral.discoverCharacteristics([BLEConstants.batteryCharUUID], for: service)
            }
        }
    }

    nonisolated func peripheral(
        _ peripheral: CBPeripheral,
        didDiscoverCharacteristicsFor service: CBService,
        error: Error?
    ) {
        guard let chars = service.characteristics else { return }
        for char in chars {
            if char.uuid == BLEConstants.dataCharUUID {
                Task { @MainActor in self.dataChar = char }
                peripheral.setNotifyValue(true, for: char)
            } else if char.uuid == BLEConstants.batteryCharUUID {
                peripheral.readValue(for: char)
                peripheral.setNotifyValue(true, for: char) // 电量变化时自动通知
            }
        }
    }

    nonisolated func peripheral(
        _ peripheral: CBPeripheral,
        didUpdateValueFor characteristic: CBCharacteristic,
        error: Error?
    ) {
        guard let data = characteristic.value, !data.isEmpty else { return }
        if characteristic.uuid == BLEConstants.batteryCharUUID {
            let level = Int(data[0])
            Task { @MainActor in self.batteryLevel = level }
        } else {
            Task { @MainActor in self.handleNotification(data) }
        }
    }
}

// MARK: - EMG Ring Buffer

final class EMGRingBuffer {
    private var buffer: [[Double]]
    private var head = 0
    private(set) var count = 0
    let capacity: Int

    init(capacity: Int) {
        self.capacity = capacity
        self.buffer = Array(repeating: [], count: capacity)
    }

    func append(_ values: [Double]) {
        buffer[head] = values
        head = (head + 1) % capacity
        count = min(count + 1, capacity)
    }

    func reset() {
        buffer = Array(repeating: [], count: capacity)
        head = 0
        count = 0
    }

    var nChannels: Int {
        buffer.first(where: { !$0.isEmpty })?.count ?? 8
    }

    /// 全缓冲区 RMS，用于特征提取 / 推理等需要长时序的场景
    func rms() -> [Double] {
        rms(window: count)
    }

    /// 短窗口 RMS，用于实时可视化 — 只取最近 `window` 帧，信号响应更灵敏
    func rms(window: Int) -> [Double] {
        let nCh = nChannels
        let w = min(max(window, 1), count)
        guard w > 0 else { return Array(repeating: 0, count: nCh) }
        var sums = [Double](repeating: 0, count: nCh)
        let start = (head - w + capacity) % capacity
        for i in 0..<w {
            let row = buffer[(start + i) % capacity]
            for ch in 0..<min(row.count, nCh) {
                sums[ch] += row[ch] * row[ch]
            }
        }
        return sums.map { sqrt($0 / Double(w)) }
    }

    func channelTimeSeries() -> [[Double]] {
        let nCh = nChannels
        guard count > 0 else { return [] }
        var result = [[Double]](repeating: [], count: nCh)
        let start = (head - count + capacity) % capacity
        for i in 0..<count {
            let row = buffer[(start + i) % capacity]
            for ch in 0..<min(row.count, nCh) {
                result[ch].append(row[ch])
            }
        }
        return result
    }
}
