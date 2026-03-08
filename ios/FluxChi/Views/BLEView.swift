import SwiftUI

struct BLEView: View {
    @EnvironmentObject var bleManager: BLEManager

    var body: some View {
        NavigationStack {
            List {
                connectionSection
                devicesSection
                dataSection
            }
            .navigationTitle("蓝牙")
            .toolbar {
                if bleManager.isConnected {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("断开", role: .destructive) {
                            bleManager.disconnect()
                        }
                    }
                }
            }
        }
    }

    // MARK: - Connection Status

    private var connectionSection: some View {
        Section {
            HStack {
                Label {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(bleManager.isConnected
                             ? (bleManager.connectedDeviceName ?? "已连接")
                             : "未连接")
                            .fontWeight(.medium)
                        Text(bleManager.isConnected
                             ? "EMG \(bleManager.emgFrameCount) 帧"
                             : "扫描并连接 WAVELETECH 手环")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } icon: {
                    Image(systemName: bleManager.isConnected
                          ? "antenna.radiowaves.left.and.right"
                          : "antenna.radiowaves.left.and.right.slash")
                        .foregroundStyle(bleManager.isConnected ? .green : .secondary)
                        .symbolEffect(.pulse, isActive: bleManager.isConnected)
                }

                Spacer()

                Circle()
                    .fill(bleManager.isConnected ? .green : .red)
                    .frame(width: 10, height: 10)
            }
        } header: {
            Text("连接状态")
        }
    }

    // MARK: - Device Discovery

    private var devicesSection: some View {
        Section {
            if bleManager.isScanning {
                HStack {
                    ProgressView()
                    Text("正在扫描…")
                        .foregroundStyle(.secondary)
                        .padding(.leading, 8)
                }
            }

            ForEach(bleManager.discoveredDevices, id: \.identifier) { device in
                Button {
                    bleManager.connect(to: device)
                } label: {
                    Label {
                        Text(device.name ?? "Unknown")
                    } icon: {
                        Image(systemName: "waveform.badge.plus")
                    }
                }
                .disabled(bleManager.isConnected)
            }

            if !bleManager.isScanning && bleManager.discoveredDevices.isEmpty {
                ContentUnavailableView {
                    Label("未发现设备", systemImage: "magnifyingglass")
                } description: {
                    Text("确保手环已开机且 USB 接收器已拔出")
                }
            }
        } header: {
            HStack {
                Text("设备")
                Spacer()
                Button {
                    bleManager.startScan()
                } label: {
                    Label("扫描", systemImage: "arrow.clockwise")
                        .font(.caption)
                }
                .disabled(bleManager.isScanning || bleManager.isConnected)
            }
        }
    }

    // MARK: - Live Data

    @ViewBuilder
    private var dataSection: some View {
        if bleManager.isConnected {
            Section("实时信号") {
                HStack(alignment: .bottom, spacing: 4) {
                    let maxVal = max(bleManager.latestRMS.max() ?? 1, 1)
                    ForEach(0..<8, id: \.self) { idx in
                        let val = bleManager.latestRMS[idx]
                        VStack(spacing: 2) {
                            RoundedRectangle(cornerRadius: 2)
                                .fill(.red.gradient)
                                .frame(height: max(4, CGFloat(val / maxVal) * 50))
                                .animation(.easeOut(duration: 0.15), value: val)

                            Text("C\(idx + 1)")
                                .font(.system(size: 8, design: .monospaced))
                                .foregroundStyle(.tertiary)
                        }
                        .frame(maxWidth: .infinity)
                    }
                }
                .frame(height: 70)
                .padding(.vertical, 4)
            }

            Section("协议信息") {
                InfoRow(label: "Service UUID", value: "974CBE30-…")
                InfoRow(label: "Data Char", value: "974CBE31-…")
                InfoRow(label: "帧格式", value: "20B (AA/BB)")
                InfoRow(label: "通道数", value: "6 × 24bit")
            }
        }
    }
}

private struct InfoRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.system(size: 13, design: .monospaced))
                .foregroundStyle(.primary)
        }
    }
}

#Preview {
    BLEView()
        .environmentObject(BLEManager())
}
