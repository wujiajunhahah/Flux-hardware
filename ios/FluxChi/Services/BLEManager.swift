import Foundation
import CoreBluetooth
import Combine

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
    static let channelCount  = 6
    /// 脱落检测：RMS 阈值（所有通道低于此值视为极低信号）
    static let detachRMSThreshold: Double = 0.5
    /// 脱落检测：需持续低信号的帧数（约 30 秒，50帧/s × 30s ÷ 每50帧检测一次 = 30次）
    static let detachConsecutiveCount = 30
}

// MARK: - Notification Names

extension Notification.Name {
    static let bleDeviceDetached = Notification.Name("BLEManager.deviceDetached")
}

@MainActor
final class BLEManager: NSObject, ObservableObject {

    // MARK: - Published

    @Published var peripheralState: CBPeripheralState = .disconnected
    @Published var discoveredDevices: [CBPeripheral] = []
    @Published var deviceRSSI: [UUID: Int] = [:]
    @Published var connectedDeviceName: String?
    @Published var isScanning = false
    @Published var latestRMS: [Double] = Array(repeating: 0, count: 6)
    @Published var emgFrameCount: Int = 0
    @Published var activeChannelCount: Int = 6
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
        guard central.state == .poweredOn else { return }
        discoveredDevices.removeAll()
        isScanning = true
        central.scanForPeripherals(
            withServices: nil,
            options: [CBCentralManagerScanOptionAllowDuplicatesKey: false]
        )
        Task {
            try? await Task.sleep(for: .seconds(15))
            if isScanning { stopScan() }
        }
    }

    func stopScan() {
        central.stopScan()
        isScanning = false
    }

    // MARK: - Connect / Disconnect

    func connect(to peripheral: CBPeripheral) {
        stopScan()
        self.peripheral = peripheral
        central.connect(peripheral, options: nil)
    }

    func disconnect() {
        guard let p = peripheral else { return }
        if let c = dataChar { p.setNotifyValue(false, for: c) }
        central.cancelPeripheralConnection(p)
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

        if emgFrameCount % 50 == 0 {
            latestRMS = emgBuffer.rms()
            checkDetachment()
            buildAndPushState()
        }
    }

    private func buildAndPushState() {
        let rms = latestRMS
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
            )
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

    private static func decodeSigned24(_ b1: Int, _ b2: Int, _ b3: Int) -> Double {
        var value = (b1 << 16) | (b2 << 8) | b3
        if value & 0x800000 != 0 { value -= 1 << 24 }
        return Double(value) / 8_388_607.0 * 4.5 / 1200.0 * 1_000_000.0
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
            }
        }
    }

    nonisolated func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        Task { @MainActor in
            self.peripheralState = .connected
            self.connectedDeviceName = peripheral.name ?? "WAVELETECH"
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
            self.peripheralState = .disconnected
            self.connectedDeviceName = nil
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
        guard let data = characteristic.value else { return }
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

    var nChannels: Int {
        buffer.first(where: { !$0.isEmpty })?.count ?? 6
    }

    func rms() -> [Double] {
        let nCh = nChannels
        guard count > 0 else { return Array(repeating: 0, count: nCh) }
        var sums = [Double](repeating: 0, count: nCh)
        let start = (head - count + capacity) % capacity
        for i in 0..<count {
            let row = buffer[(start + i) % capacity]
            for ch in 0..<min(row.count, nCh) {
                sums[ch] += row[ch] * row[ch]
            }
        }
        return sums.map { sqrt($0 / Double(count)) }
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
