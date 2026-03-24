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

    // MARK: - Init

    override init() {
        super.init()
        central = CBCentralManager(delegate: nil, queue: nil)
        central.delegate = self
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
        if let c = dataChar { p.setNotifyValue(false, for: c) }
        central.cancelPeripheralConnection(p)
        let deviceName = p.name ?? connectedDeviceName ?? "未知设备"
        FluxLog.ble.info("主动断开: \(deviceName)")
        peripheral = nil
        dataChar = nil
        connectedDeviceName = nil
        peripheralState = .disconnected
        staminaEngine.reset()
    }

    // MARK: - EMG Parsing

    private func handleNotification(_ data: Data) {
        guard data.count >= 20, data[0] == BLEConstants.emgFlag else { return }

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

        if emgFrameCount % 8 == 0 {
            latestRMS = emgBuffer.rms(window: 24)
        }

        if emgFrameCount % 50 == 0 {
            checkDetachment()
            buildAndPushState()

            let rmsPreview = latestRMS.prefix(3).map { String(format: "%.1f", $0) }.joined(separator: ", ")
            FluxLog.ble.debug("Frame \(emgFrameCount) | RMS: [\(rmsPreview), ...] | Channels: \(nCh)")
        }
    }

    private func buildAndPushState() {
        let rms = emgBuffer.rms()
        let now = Date().timeIntervalSince1970
        let rawChannels = emgBuffer.channelTimeSeries()

        let prediction = classifier.predict(channels: rawChannels)
        let classifiedActivity = prediction?.label

        let r = staminaEngine.update(rms: rms, rawChannels: rawChannels, timestamp: now, classifiedActivity: classifiedActivity)

        let activity = classifiedActivity ?? (r.isWorking ? "working" : "rest")
        let confidence = prediction?.confidence ?? 1.0
        let probs = prediction?.probabilities ?? [activity: 1.0]

        let state = FluxState(
            timestamp: now,
            activity: activity,
            confidence: confidence,
            probabilities: probs,
            rms: rms,
            emgSampleCount: emgFrameCount,
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
        onStateUpdate?(state)
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
            self.peripheral = nil
            self.dataChar = nil
            self.connectedDeviceName = nil
            self.peripheralState = .disconnected
            self.emgBuffer.reset()
            self.emgFrameCount = 0
            self.lowSignalConsecutive = 0
            self.detachNotified = false
            self.latestRMS = Array(repeating: 0, count: 8)
            self.staminaEngine.reset()
            if let error = error {
                FluxLog.ble.error("意外断开: \(deviceName) — 状态已完整重置", error: error)
            } else {
                FluxLog.ble.info("正常断开: \(deviceName)")
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
