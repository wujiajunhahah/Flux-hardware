import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var service: FluxService
    @EnvironmentObject var bleManager: BLEManager
    @EnvironmentObject var personalization: PersonalizationManager

    @State private var editHost: String = ""
    @State private var editPort: String = ""
    @State private var showResetAlert = false
    @State private var showLogExportOptions = false
    @State private var showClearLogsAlert = false
    @FocusState private var focusedField: Field?

    private enum Field { case host, port }

    var body: some View {
        NavigationStack {
            Form {
                bleSection
                serverSection
                statusSection
                personalizationSection
                logsSection
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
            .sheet(isPresented: $showLogExportOptions) {
                logExportSheet
            }
            .alert("清空日志", isPresented: $showClearLogsAlert) {
                Button("取消", role: .cancel) {}
                Button("清空", role: .destructive) {
                    FluxLogger.shared.clearLogs()
                }
            } message: {
                Text("将清空所有日志记录，此操作不可撤销")
            }
        }
    }

    // MARK: - BLE

    private var bleSection: some View {
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

            if bleManager.isConnected {
                Button("断开蓝牙", role: .destructive) {
                    bleManager.disconnect()
                }
            } else {
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
                        Label(device.name ?? "Unknown", systemImage: "waveform.badge.plus")
                    }
                }

                Button {
                    bleManager.startScan()
                } label: {
                    Label("扫描设备", systemImage: "arrow.clockwise")
                }
                .disabled(bleManager.isScanning)
            }
        } header: {
            Text("蓝牙直连")
        } footer: {
            Text("BLE 连接时自动停止 WiFi 轮询")
        }
    }

    // MARK: - Server

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
                    .onSubmit { focusedField = .port }
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
            Text("WiFi 服务器")
        } footer: {
            Text("手机和电脑需在同一 WiFi 下")
        }
    }

    // MARK: - Status

    @ViewBuilder
    private var statusSection: some View {
        Section("状态") {
            StatusRow(
                icon: "circle.fill",
                label: "连接",
                value: service.isConnected ? "已连接" : (bleManager.isConnected ? "BLE" : "未连接"),
                tint: (service.isConnected || bleManager.isConnected) ? .green : .red
            )

            if let status = service.serverStatus {
                StatusRow(icon: "brain", label: "模型",
                          value: status.modelLoaded ? "已加载" : "未加载",
                          tint: status.modelLoaded ? .green : .orange)
                StatusRow(icon: "gauge.with.dots.needle.33percent", label: "模式",
                          value: status.demoMode ? "演示" : "实时",
                          tint: status.demoMode ? .orange : .blue)
                if let uptime = status.uptimeSec {
                    StatusRow(icon: "clock", label: "运行",
                              value: formatUptime(uptime), tint: .secondary)
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

    // MARK: - Personalization

    private var personalizationSection: some View {
        Section {
            HStack {
                Label("学习次数", systemImage: "brain.head.profile")
                Spacer()
                Text("\(personalization.trainingCount)")
                    .foregroundStyle(.secondary)
            }

            if personalization.trainingCount > 0 {
                HStack {
                    Label("校准偏移", systemImage: "slider.horizontal.3")
                    Spacer()
                    Text(String(format: "%+.1f", personalization.calibrationOffset))
                        .font(Flux.Typography.mono)
                        .foregroundStyle(.secondary)
                }

                if personalization.estimatedAccuracy > 0 {
                    HStack {
                        Label("估计准确度", systemImage: "target")
                        Spacer()
                        Text("\(Int(personalization.estimatedAccuracy))%")
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if let date = personalization.lastTrainedAt {
                HStack {
                    Label("上次训练", systemImage: "clock.arrow.circlepath")
                    Spacer()
                    Text(date, style: .relative)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if personalization.isTraining {
                HStack {
                    ProgressView()
                    Text("模型训练中…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.leading, 8)
                }
            }
        } header: {
            Text("个性化")
        } footer: {
            Text("每次记录反馈后，模型会自动学习你的个人特征")
        }
    }

    // MARK: - Logs

    private var logsSection: some View {
        Section("日志") {
            HStack {
                Label("日志条目", systemImage: "doc.text")
                Spacer()
                Text("\(FluxLogger.shared.entries.count)")
                    .foregroundStyle(.secondary)
            }

            NavigationLink {
                LogViewerView()
            } label: {
                Label("查看日志", systemImage: "list.bullet")
            }

            Button {
                showLogExportOptions = true
            } label: {
                Label("导出日志", systemImage: "square.and.arrow.up")
            }

            Button {
                showClearLogsAlert = true
            } label: {
                Label("清空日志", systemImage: "trash")
                    .foregroundStyle(.red)
            }
        } footer: {
            Text("日志用于调试和分析问题")
        }
    }

    @ViewBuilder
    private var logExportSheet: some View {
        NavigationStack {
            Form {
                Section("导出格式") {
                    Button("导出为 JSON") {
                        exportLogsAsJSON()
                    }
                    Button("导出为文本") {
                        exportLogsAsText()
                    }
                }

                Section("筛选选项") {
                    Button("仅导出错误日志") {
                        exportErrorLogs()
                    }
                    Button("仅导出 BLE 日志") {
                        exportBLELogs()
                    }
                    Button("导出全部日志") {
                        exportAllLogs()
                    }
                }
            }
            .navigationTitle("导出日志")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("取消") {
                        showLogExportOptions = false
                    }
                }
            }
        }
    }

    // MARK: - Log Export Actions

    private func exportLogsAsJSON() {
        Task {
            do {
                let url = try ExportManager.shareLogsURL()
                showLogExportOptions = false
                // TODO: 展示分享表
                FluxLogger.shared.info("日志已导出: \(url.lastPathComponent)", category: .export)
            } catch {
                FluxLogger.shared.error("导出失败", category: .export, error: error)
            }
        }
    }

    private func exportLogsAsText() {
        Task {
            do {
                let url = try ExportManager.shareLogsTextURL()
                showLogExportOptions = false
                FluxLogger.shared.info("日志已导出: \(url.lastPathComponent)", category: .export)
            } catch {
                FluxLogger.shared.error("导出失败", category: .export, error: error)
            }
        }
    }

    private func exportErrorLogs() {
        Task {
            do {
                let url = try ExportManager.shareLogsURL(options: .errorsOnly)
                showLogExportOptions = false
                FluxLogger.shared.info("错误日志已导出: \(url.lastPathComponent)", category: .export)
            } catch {
                FluxLogger.shared.error("导出失败", category: .export, error: error)
            }
        }
    }

    private func exportBLELogs() {
        Task {
            do {
                let url = try ExportManager.shareLogsURL(options: .bleOnly)
                showLogExportOptions = false
                FluxLogger.shared.info("BLE 日志已导出: \(url.lastPathComponent)", category: .export)
            } catch {
                FluxLogger.shared.error("导出失败", category: .export, error: error)
            }
        }
    }

    private func exportAllLogs() {
        Task {
            do {
                let url = try ExportManager.shareLogsURL(options: .default)
                showLogExportOptions = false
                FluxLogger.shared.info("全部日志已导出: \(url.lastPathComponent)", category: .export)
            } catch {
                FluxLogger.shared.error("导出失败", category: .export, error: error)
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
        }
    }

    // MARK: - About

    private var aboutSection: some View {
        Section("关于") {
            HStack {
                Text(Flux.App.name)
                    .fontWeight(.medium)
                Spacer()
                Text("v\(Flux.App.version) · EMG Stamina Engine")
                    .foregroundStyle(.secondary)
                    .font(.caption)
            }

            Link(destination: Flux.App.githubURL) {
                Label("GitHub", systemImage: "link")
            }
        }
    }

    // MARK: - Helpers

    private func applyServerConfig() {
        service.host = editHost
        if let p = Int(editPort), p > 0, p <= 65535 {
            service.port = p
        }
        service.reconnect()
    }

    private func formatUptime(_ seconds: Double) -> String {
        Flux.formatDuration(seconds)
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
