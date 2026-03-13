import SwiftUI

/// 极客数据面板 — 从设置进入，展示 EMG 原始信号和调试数据
struct GeekDataPanel: View {
    @EnvironmentObject var service: FluxService
    @EnvironmentObject var bleManager: BLEManager

    var body: some View {
        List {
            emgRMSSection
            bleSignalSection
            protocolSection
            calibrationSection
        }
        .navigationTitle("极客数据")
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - EMG RMS (WiFi)

    @ViewBuilder
    private var emgRMSSection: some View {
        if let rms = service.state?.rms, !rms.isEmpty {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    FluxEMGBars(rms: rms, height: 60)
                        .drawingGroup()
                }
                .padding(.vertical, 4)

                HStack {
                    Text("通道数")
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(rms.count)")
                        .font(.system(size: 13, design: .monospaced))
                }

                if let maxVal = rms.max() {
                    HStack {
                        Text("峰值 RMS")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(String(format: "%.2f", maxVal))
                            .font(.system(size: 13, design: .monospaced))
                    }
                }
            } header: {
                Text("EMG RMS（WiFi）")
            }
        }
    }

    // MARK: - BLE Signal

    @ViewBuilder
    private var bleSignalSection: some View {
        if bleManager.isConnected {
            Section {
                HStack(alignment: .bottom, spacing: 4) {
                    let rms = bleManager.latestRMS
                    let channelCount = rms.count
                    let maxVal = max(rms.max() ?? 1, 1)
                    ForEach(0..<channelCount, id: \.self) { idx in
                        let val = rms[idx]
                        VStack(spacing: 2) {
                            RoundedRectangle(cornerRadius: 2)
                                .fill(Color(.systemIndigo).gradient)
                                .frame(height: max(4, CGFloat(val / maxVal) * 60))
                                .animation(.easeOut(duration: 0.15), value: val)

                            Text("C\(idx + 1)")
                                .font(.system(size: 8, design: .monospaced))
                                .foregroundStyle(.tertiary)
                        }
                        .frame(maxWidth: .infinity)
                    }
                }
                .frame(height: 80)
                .padding(.vertical, 4)

                HStack {
                    Text("帧计数")
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(bleManager.emgFrameCount)")
                        .font(.system(size: 13, design: .monospaced))
                }
            } header: {
                Text("BLE 实时信号")
            }
        } else {
            Section {
                ContentUnavailableView {
                    Label("BLE 未连接", systemImage: "antenna.radiowaves.left.and.right.slash")
                } description: {
                    Text("连接 WAVELETECH 手环后可查看实时信号")
                }
            } header: {
                Text("BLE 实时信号")
            }
        }
    }

    // MARK: - Protocol

    private var protocolSection: some View {
        Section("协议信息") {
            geekRow("Service UUID", "974CBE30-…")
            geekRow("Data Char", "974CBE31-…")
            geekRow("帧格式", "20B (AA/BB)")
            geekRow("通道数", "6 × 24bit")
        }
    }

    // MARK: - Calibration Debug

    private var calibrationSection: some View {
        Section {
            let last = UserDefaults.standard.double(forKey: "flux_last_calibration")
            if last > 0 {
                let date = Date(timeIntervalSince1970: last)
                geekRow("上次校准", date.formatted(.dateTime.month().day().hour().minute()))
            } else {
                geekRow("上次校准", "从未校准")
            }

            geekRow("服务器", "\(service.host):\(service.port)")
            geekRow("WiFi 连接", service.isConnected ? "已连接" : "未连接")
            geekRow("BLE 连接", bleManager.isConnected ? "已连接" : "未连接")
        } header: {
            Text("调试信息")
        }
    }

    private func geekRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.system(size: 13, design: .monospaced))
        }
    }
}

#Preview {
    NavigationStack {
        GeekDataPanel()
            .environmentObject(FluxService())
            .environmentObject(BLEManager())
    }
}
