import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var service: FluxService

    @State private var editHost: String = ""
    @State private var editPort: String = ""
    @State private var showResetAlert = false
    @FocusState private var focusedField: Field?

    private enum Field { case host, port }

    var body: some View {
        NavigationStack {
            Form {
                serverSection
                statusSection
                controlSection
                aboutSection
            }
            .navigationTitle("设置")
            .scrollDismissesKeyboard(.interactively)
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("完成") {
                        focusedField = nil
                        applyServerConfig()
                    }
                    .fontWeight(.semibold)
                }
            }
            .onAppear {
                editHost = service.host
                editPort = "\(service.port)"
            }
        }
    }

    private var serverSection: some View {
        Section {
            HStack {
                Image(systemName: "globe")
                    .foregroundStyle(.secondary)
                TextField("主机地址", text: $editHost)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.URL)
                    .focused($focusedField, equals: .host)
                    .onSubmit {
                        focusedField = .port
                    }
            }

            HStack {
                Image(systemName: "number")
                    .foregroundStyle(.secondary)
                TextField("端口", text: $editPort)
                    .keyboardType(.numberPad)
                    .focused($focusedField, equals: .port)
            }

            Button {
                focusedField = nil
                applyServerConfig()
            } label: {
                HStack {
                    Text("应用并连接")
                    Spacer()
                    if service.isConnected {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    }
                }
            }
        } header: {
            Text("服务器")
        } footer: {
            Text("手机和电脑需在同一 WiFi 下")
        }
    }

    @ViewBuilder
    private var statusSection: some View {
        Section("状态") {
            StatusRow(
                icon: "circle.fill",
                label: "连接",
                value: service.isConnected ? "已连接" : "未连接",
                tint: service.isConnected ? .green : .red
            )

            if let status = service.serverStatus {
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
                        label: "运行",
                        value: formatUptime(uptime),
                        tint: .secondary
                    )
                }
            }

            if let err = service.connectionError {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(err)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

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
        }
    }

    private var aboutSection: some View {
        Section("关于") {
            HStack {
                Text("FluxChi")
                    .fontWeight(.medium)
                Spacer()
                Text("EMG Stamina Engine")
                    .foregroundStyle(.secondary)
                    .font(.caption)
            }

            Link(destination: URL(string: "https://github.com/wujiajunhahah/formydegree")!) {
                Label("GitHub", systemImage: "link")
            }
        }
    }

    private func applyServerConfig() {
        service.host = editHost
        if let p = Int(editPort), p > 0, p <= 65535 {
            service.port = p
        }
        service.reconnect()
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
