import Foundation
import CoreBluetooth
import Combine

/// CoreBluetooth manager for direct WAVELETECH wristband connection.
@MainActor
final class BLEManager: NSObject, ObservableObject {

    // MARK: - Constants (reverse-engineered from WAVELETECH protocol)

    static let serviceUUID    = CBUUID(string: "974CBE30-3E83-465E-ACDE-6F92FE712134")
    static let dataCharUUID   = CBUUID(string: "974CBE31-3E83-465E-ACDE-6F92FE712134")
    static let writeCharUUID  = CBUUID(string: "974CBE32-3E83-465E-ACDE-6F92FE712134")
    static let devicePrefix   = "WL"

    private static let emgFlag: UInt8 = 0xAA
    private static let imuFlag: UInt8 = 0xBB
    private static let channelCount = 8

    // MARK: - Published

    @Published var peripheralState: CBPeripheralState = .disconnected
    @Published var discoveredDevices: [CBPeripheral] = []
    @Published var connectedDeviceName: String?
    @Published var isScanning = false
    @Published var latestRMS: [Double] = Array(repeating: 0, count: 8)
    @Published var emgFrameCount: Int = 0

    var isConnected: Bool { peripheralState == .connected }

    // MARK: - Callbacks

    var onEMGSample: (([Double]) -> Void)?

    // MARK: - Private

    private var central: CBCentralManager!
    private var peripheral: CBPeripheral?
    private var dataChar: CBCharacteristic?

    private let emgBuffer = EMGRingBuffer(capacity: 250)

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
            withServices: [Self.serviceUUID],
            options: [CBCentralManagerScanOptionAllowDuplicatesKey: false]
        )

        Task {
            try? await Task.sleep(for: .seconds(10))
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
        if let c = dataChar {
            p.setNotifyValue(false, for: c)
        }
        central.cancelPeripheralConnection(p)
        peripheral = nil
        dataChar = nil
        connectedDeviceName = nil
        peripheralState = .disconnected
    }

    // MARK: - EMG Parsing

    private func handleNotification(_ data: Data) {
        guard data.count >= 20 else { return }

        let frameType = data[0]

        if frameType == Self.emgFlag {
            var values = [Double](repeating: 0, count: Self.channelCount)
            let payload = data.dropFirst(2) // skip type + seq
            let nChannels = min(payload.count / 3, Self.channelCount)

            for i in 0..<nChannels {
                let offset = i * 3
                let b1 = Int(payload[payload.startIndex + offset])
                let b2 = Int(payload[payload.startIndex + offset + 1])
                let b3 = Int(payload[payload.startIndex + offset + 2])
                values[i] = Self.decodeSigned24(b1, b2, b3)
            }

            emgBuffer.append(values)
            emgFrameCount += 1
            onEMGSample?(values)

            if emgFrameCount % 50 == 0 {
                latestRMS = emgBuffer.rms()
            }
        }
    }

    private static func decodeSigned24(_ b1: Int, _ b2: Int, _ b3: Int) -> Double {
        var value = (b1 << 16) | (b2 << 8) | b3
        if value & 0x800000 != 0 {
            value -= 1 << 24
        }
        let volts = Double(value) / 8_388_607.0 * 4.5
        return (volts / 1200.0) * 1_000_000.0
    }
}

// MARK: - CBCentralManagerDelegate

extension BLEManager: CBCentralManagerDelegate {

    nonisolated func centralManagerDidUpdateState(_ central: CBCentralManager) {
        Task { @MainActor in
            if central.state == .poweredOn {
                // Ready to scan
            }
        }
    }

    nonisolated func centralManager(
        _ central: CBCentralManager,
        didDiscover peripheral: CBPeripheral,
        advertisementData: [String: Any],
        rssi RSSI: NSNumber
    ) {
        Task { @MainActor in
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
            peripheral.discoverServices([Self.serviceUUID])
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
        for service in services where service.uuid == Self.serviceUUID {
            peripheral.discoverCharacteristics(
                [Self.dataCharUUID, Self.writeCharUUID],
                for: service
            )
        }
    }

    nonisolated func peripheral(
        _ peripheral: CBPeripheral,
        didDiscoverCharacteristicsFor service: CBService,
        error: Error?
    ) {
        guard let chars = service.characteristics else { return }
        for char in chars {
            if char.uuid == Self.dataCharUUID {
                Task { @MainActor in self.dataChar = char }
                peripheral.setNotifyValue(true, for: char)
            }
        }
    }

    nonisolated func peripheral(
        _ peripheral: CBPeripheral,
        didUpdateValueFor characteristic: CBCharacteristic,
        error: Error?
    ) {
        guard let data = characteristic.value else { return }
        Task { @MainActor in
            self.handleNotification(data)
        }
    }
}

// MARK: - EMG Ring Buffer

private final class EMGRingBuffer {
    private var buffer: [[Double]]
    private var head = 0
    private var count = 0
    private let capacity: Int

    init(capacity: Int) {
        self.capacity = capacity
        self.buffer = Array(repeating: [], count: capacity)
    }

    func append(_ values: [Double]) {
        buffer[head] = values
        head = (head + 1) % capacity
        count = min(count + 1, capacity)
    }

    func rms() -> [Double] {
        guard count > 0 else { return Array(repeating: 0, count: 8) }
        let nCh = buffer.first(where: { !$0.isEmpty })?.count ?? 8
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
}
