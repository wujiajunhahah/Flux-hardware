import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var service: FluxService

    @State private var editHost: String = ""
    @State private var editPort: String = ""
    @State private var showResetAlert = false

    var body: some View {
        NavigationStack {
            Form {
                serverSection
                statusSection
                controlSection
                aboutSection
            }
            .navigationTitle("设置")
            .onAppear {
                editHost = service.host
                editPort = "\(service.port)"
            }
        }
    }

    // MARK: - Server

    private var serverSection: some View {
        Section("服务器") {
            HStack {
                Image(systemName: "globe")
                    .foregroundStyle(.secondary)
                TextField("主机地址", text: $editHost)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.URL)
                    .onSubmit { applyServerConfig() }
            }

            HStack {
                Image(systemName: "number")
                    .foregroundStyle(.secondary)
                TextField("端口", text: $editPort)
                    .keyboardType(.numberPad)
                    .onSubmit { applyServerConfig() }
            }

            Button("应用") {
                applyServerConfig()
            }
            .disabled(editHost == service.host && editPort == "\(service.port)")
        }
    }

    // MARK: - Status

    @ViewBuilder
    private var statusSection: some View {
        if let status = service.serverStatus {
            Section("服务器状态") {
                StatusRow(
                    icon: "circle.fill",
                    label: "连接",
                    value: status.connected ? "正常" : "断开",
                    tint: status.connected ? .green : .red
                )
                StatusRow(
                    icon: "brain",
                    label: "模型",
                    value: status.modelLoaded ? "已加载" : "未加载",
                    tint: status.modelLoaded ? .green : .orange
                )
                StatusRow(
                    icon: "gauge.with.dots.needle.33percent",
                    label: "模式",
                    value: status.demoMode ? "演示" : "实时",
                    tint: status.demoMode ? .orange : .blue
                )
                StatusRow(
                    icon: "speedometer",
                    label: "速度",
                    value: "\(String(format: "%.1f", status.speed))×",
                    tint: .secondary
                )
                if let uptime = status.uptimeSec {
                    StatusRow(
                        icon: "clock",
                        label: "运行时间",
                        value: formatUptime(uptime),
                        tint: .secondary
                    )
                }
            }
        }
    }

    // MARK: - Control

    private var controlSection: some View {
        Section("控制") {
            Button {
                showResetAlert = true
            } label: {
                Label("重置续航值", systemImage: "arrow.counterclockwise")
            }
            .alert("重置续航", isPresented: $showResetAlert) {
                Button("取消", role: .cancel) {}
                Button("重置", role: .destructive) {
                    Task { await service.resetStamina() }
                }
            } message: {
                Text("将续航值重置为 100%")
            }

            Button {
                Task { await service.saveStamina() }
            } label: {
                Label("保存状态", systemImage: "square.and.arrow.down")
            }

            Button {
                Task { await service.fetchStatus() }
            } label: {
                Label("刷新状态", systemImage: "arrow.clockwise")
            }
        }
    }

    // MARK: - About

    private var aboutSection: some View {
        Section("关于") {
            HStack {
                Text("FluxChi")
                    .fontWeight(.medium)
                Spacer()
                Text("EMG Stamina Engine")
                    .foregroundStyle(.secondary)
            }

            HStack {
                Text("协议")
                Spacer()
                Text("REST / SSE / BLE")
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            Link(destination: URL(string: "https://github.com/wujiajunhahah/formydegree")!) {
                Label("GitHub 仓库", systemImage: "link")
            }
        }
    }

    // MARK: - Helpers

    private func applyServerConfig() {
        service.host = editHost
        if let p = Int(editPort), p > 0, p <= 65535 {
            service.port = p
        }
        service.stopSSE()
        service.startSSE()
        Task { await service.fetchStatus() }
    }

    private func formatUptime(_ seconds: Double) -> String {
        let h = Int(seconds) / 3600
        let m = (Int(seconds) % 3600) / 60
        let s = Int(seconds) % 60
        if h > 0 { return "\(h)h \(m)m" }
        if m > 0 { return "\(m)m \(s)s" }
        return "\(s)s"
    }
}

private struct StatusRow: View {
    let icon: String
    let label: String
    let value: String
    let tint: Color

    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundStyle(tint)
                .frame(width: 20)
            Text(label)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
        }
    }
}

#Preview {
    SettingsView()
        .environmentObject(FluxService())
}
