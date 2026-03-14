import SwiftUI

/// 极简连接页 — 打开即扫描，点击即连接，纯原生动画
struct ConnectionGuideSheet: View {
    @EnvironmentObject var service: FluxService
    @EnvironmentObject var bleManager: BLEManager
    @Environment(\.dismiss) private var dismiss

    @State private var showWiFiInput = false
    @State private var editHost: String = ""
    @State private var editPort: String = ""

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Spacer()
                scanIndicator
                Spacer()
                deviceList
                    .padding(.bottom, 24)
                wifiToggle
                    .padding(.bottom, 16)
            }
            .padding(.horizontal, 24)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("关闭") { dismiss() }
                }
            }
            .onAppear {
                bleManager.startScan()
                editHost = service.host
                editPort = "\(service.port)"
            }
            .onChange(of: bleManager.isConnected) { _, connected in
                if connected {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                        dismiss()
                    }
                }
            }
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
        .presentationBackground(.regularMaterial)
    }

    // MARK: - Scan Indicator（纯原生 symbolEffect）

    private var scanIndicator: some View {
        VStack(spacing: 16) {
            Image(systemName: bleManager.isConnected
                  ? "checkmark.circle.fill"
                  : "antenna.radiowaves.left.and.right")
                .font(.system(size: 56))
                .foregroundStyle(bleManager.isConnected ? .green : Flux.Colors.accent)
                .symbolEffect(.variableColor.iterative,
                              isActive: bleManager.isScanning && !bleManager.isConnected)
                .contentTransition(.symbolEffect(.replace))

            Text(statusText)
                .font(.title3.weight(.semibold))

            if bleManager.isScanning && bleManager.discoveredDevices.isEmpty {
                Text("请确保手环已开机并在附近")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var statusText: String {
        if bleManager.isConnected {
            return "已连接"
        } else if bleManager.isScanning {
            return bleManager.discoveredDevices.isEmpty ? "正在搜索…" : "发现设备"
        } else {
            return "蓝牙扫描"
        }
    }

    // MARK: - Device List

    @ViewBuilder
    private var deviceList: some View {
        if !bleManager.discoveredDevices.isEmpty && !bleManager.isConnected {
            VStack(spacing: 2) {
                ForEach(Array(bleManager.discoveredDevices.enumerated()), id: \.element.identifier) { _, device in
                    Button {
                        bleManager.connect(to: device)
                    } label: {
                        HStack(spacing: 14) {
                            Image(systemName: "sensor.tag.radiowaves.forward.fill")
                                .font(.system(size: 20))
                                .foregroundStyle(Flux.Colors.accent)
                                .frame(width: 36)

                            Text(device.name ?? "未知设备")
                                .font(.body.weight(.medium))
                                .foregroundStyle(.primary)

                            Spacer()

                            Text("连接")
                                .font(.subheadline.weight(.medium))
                                .foregroundStyle(Flux.Colors.accent)
                        }
                        .padding(.vertical, 14)
                        .padding(.horizontal, 16)
                        .background(.regularMaterial, in: .rect(cornerRadius: 12))
                    }
                    .buttonStyle(.plain)
                }
            }
        } else if bleManager.isConnected {
            HStack(spacing: 10) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text(bleManager.connectedDeviceName ?? "已连接")
                    .font(.body.weight(.medium))
                Spacer()
            }
            .padding(14)
            .background(.green.opacity(0.08), in: .rect(cornerRadius: 12))
        }
    }

    // MARK: - WiFi Toggle

    private var wifiToggle: some View {
        VStack(spacing: 10) {
            Button {
                withAnimation { showWiFiInput.toggle() }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "wifi")
                        .font(.caption2)
                    Text("WiFi 中转")
                        .font(.caption)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 8, weight: .bold))
                        .rotationEffect(showWiFiInput ? .degrees(-180) : .zero)
                }
                .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)

            if showWiFiInput {
                HStack(spacing: 8) {
                    TextField("主机", text: $editHost)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                        .font(.footnote.monospaced())
                        .padding(10)
                        .background(.regularMaterial, in: .rect(cornerRadius: 8))

                    TextField("端口", text: $editPort)
                        .keyboardType(.numberPad)
                        .font(.footnote.monospaced())
                        .frame(width: 64)
                        .padding(10)
                        .background(.regularMaterial, in: .rect(cornerRadius: 8))

                    Button {
                        service.host = editHost
                        if let p = Int(editPort), p > 0, p <= 65535 {
                            service.port = p
                        }
                        service.reconnect()
                    } label: {
                        Image(systemName: "arrow.right.circle.fill")
                            .font(.title3)
                            .foregroundStyle(Flux.Colors.accent)
                    }
                    .buttonStyle(.plain)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
    }
}
