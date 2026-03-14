import SwiftUI
import SwiftData

struct DashboardView: View {
    @EnvironmentObject var service: FluxService
    @EnvironmentObject var bleManager: BLEManager
    @EnvironmentObject var sessionManager: SessionManager
    @EnvironmentObject var alertManager: AlertManager
    @EnvironmentObject var liveActivityManager: LiveActivityManager
    @Environment(\.modelContext) private var modelContext

    @Query private var todaySessions: [Session]

    @Binding var showActiveSession: Bool
    @Binding var finishedSession: Session?

    @State private var showSegmentPicker = false
    @State private var showFeedback = false
    @State private var showSummary = false   // recordButton 直接结束时用

    // Daily Insight
    @State private var dailyInsightText: String?
    @State private var isDailyInsightLoading = false

    init(showActiveSession: Binding<Bool>, finishedSession: Binding<Session?>) {
        _showActiveSession = showActiveSession
        _finishedSession = finishedSession

        let startOfDay = Calendar.current.startOfDay(for: Date())
        let predicate = #Predicate<Session> { session in
            session.startedAt >= startOfDay && session.endedAt != nil
        }
        _todaySessions = Query(filter: predicate, sort: \Session.startedAt, order: .reverse)
    }

    private var stamina: StaminaData? { service.state?.stamina }
    private var decision: DecisionData? { service.state?.decision }
    private var staminaValue: Double { service.personalizedStaminaValue }
    private var staminaState: StaminaState {
        StaminaState(rawValue: stamina?.state ?? "focused") ?? .focused
    }
    private var isLive: Bool { service.isConnected || bleManager.isConnected }

    private var isCalibratedToday: Bool {
        let last = UserDefaults.standard.double(forKey: "flux_last_calibration")
        guard last > 0 else { return false }
        return Calendar.current.isDateInToday(Date(timeIntervalSince1970: last))
    }

    enum DashCalPhase { case idle, baseline, mvc }
    @State private var calPhase: DashCalPhase = .idle
    @State private var calibrationProgress: CGFloat = 0
    @State private var calibrationTimer: Timer?
    private var isCalibrating: Bool { calPhase != .idle }

    // MARK: - Body

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    if !isLive { connectionBanner }

                    if isLive && !isCalibratedToday {
                        calibrationBanner
                    }

                    StaminaRingView(value: staminaValue, state: staminaState)
                        .drawingGroup() // GPU 离屏渲染，避免 AngularGradient 每帧重算
                        .padding(.vertical, 4)

                    if sessionManager.isRecording {
                        recordingBar
                    }

                    recommendationCard

                    // 提取为独立 struct，stamina 不变时不重绘
                    DimensionsRow(
                        consistency: stamina?.consistency ?? 0,
                        tension: stamina?.tension ?? 0,
                        fatigue: stamina?.fatigue ?? 0
                    )

                    if !todaySessions.isEmpty {
                        todaySummaryCard
                        dailyInsightCard
                    }
                }
                .padding(.horizontal)
                .padding(.bottom, 80)
            }
            .navigationTitle("FocuX")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    HStack(spacing: 6) {
                        FluxLiveIndicator(isLive: isLive)
                        if let battery = bleManager.batteryLevel {
                            HStack(spacing: 2) {
                                Image(systemName: batteryIcon(battery))
                                    .font(.system(size: 10))
                                    .foregroundStyle(battery > 20 ? .green : .red)
                                Text("\(battery)%")
                                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                ToolbarItem(placement: .topBarTrailing) { recordButton }
            }
            .onAppear {
                sessionManager.configure(
                    modelContext: modelContext,
                    stateProvider: { [weak service] in service?.state }
                )
            }
            .onReceive(NotificationCenter.default.publisher(for: FluxChiApp.showActiveSessionNotification)) { _ in
                if sessionManager.isRecording && !showActiveSession {
                    showActiveSession = true
                }
            }
            .sheet(isPresented: $showFeedback) {
                if let s = finishedSession { FeedbackView(session: s) }
            }
            .sheet(isPresented: $showSummary) {
                if let s = finishedSession { SessionSummarySheet(session: s) }
            }
        }
    }

    // MARK: - Calibration Banner

    @ViewBuilder
    private var calibrationBanner: some View {
        if isCalibrating {
            let ringColor: Color = calPhase == .mvc ? .orange : .green
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .stroke(ringColor.opacity(0.2), lineWidth: 3)
                        .frame(width: 32, height: 32)
                    Circle()
                        .trim(from: 0, to: calibrationProgress)
                        .stroke(ringColor, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                        .frame(width: 32, height: 32)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(calPhase == .mvc ? "最大握拳！" : "校准中…请放松手臂")
                        .font(.subheadline.weight(.medium))
                    Text(calPhase == .mvc ? "尽全力握拳保持 5 秒" : "保持自然姿势 10 秒")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(12)
            .background(ringColor.opacity(0.08), in: .rect(cornerRadius: 16))
        } else {
            HStack(spacing: 10) {
                Image(systemName: "tuningfork")
                    .foregroundStyle(Color(.systemTeal))
                VStack(alignment: .leading, spacing: 2) {
                    Text("今日未校准")
                        .font(.subheadline.weight(.medium))
                    Text("校准可提高续航值准确度")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("去校准") {
                    startDailyCalibration()
                }
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color(.systemTeal), in: Capsule())
            }
            .padding(12)
            .background(Color(.systemTeal).opacity(0.08), in: .rect(cornerRadius: 16))
        }
    }

    private func startDailyCalibration() {
        calPhase = .baseline
        calibrationProgress = 0

        // Phase 1: 静息基线 10 秒
        calibrationTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { timer in
            Task { @MainActor in
                calibrationProgress += 0.01
                if calibrationProgress >= 1.0 {
                    timer.invalidate()
                    calibrationTimer = nil
                    startDailyMVC()
                }
            }
        }
    }

    private func startDailyMVC() {
        calPhase = .mvc
        calibrationProgress = 0

        // Phase 2: 最大握拳 5 秒
        calibrationTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { timer in
            Task { @MainActor in
                calibrationProgress += 0.02  // 5 秒完成
                if calibrationProgress >= 1.0 {
                    timer.invalidate()
                    calibrationTimer = nil
                    calPhase = .idle
                    UserDefaults.standard.set(
                        Date().timeIntervalSince1970,
                        forKey: "flux_last_calibration"
                    )
                }
            }
        }
    }

    // MARK: - Connection

    @ViewBuilder
    private var connectionBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            Text("未连接")
                .font(.subheadline.weight(.medium))
            Spacer()
            if let err = service.connectionError {
                Text(err).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: 16))
    }

    // MARK: - Recommendation

    @ViewBuilder
    private var recommendationCard: some View {
        if let d = decision {
            let rec = Recommendation(rawValue: d.recommendation) ?? .keepWorking
            let urgencyColor = Flux.Colors.forUrgency(d.urgency)

            HStack(spacing: 14) {
                ZStack {
                    Circle()
                        .fill(urgencyColor.opacity(0.12))
                        .frame(width: 44, height: 44)
                    Image(systemName: rec.systemImage)
                        .font(.title3)
                        .foregroundStyle(urgencyColor)
                        .symbolRenderingMode(.hierarchical)
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text(rec.displayName)
                        .font(.subheadline.weight(.semibold))
                    if let reason = d.reasons.first {
                        Text(reason)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                }
                Spacer()
            }
            .padding(14)
            .background(.regularMaterial, in: .rect(cornerRadius: 16))
        }
    }

    // MARK: - Recording

    private var recordButton: some View {
        Button {
            if sessionManager.isRecording {
                liveActivityManager.endActivity()
                if let session = sessionManager.endSession() {
                    let summary = SummaryEngine.generate(for: session)
                    SummaryEngine.apply(summary, to: session)
                    try? modelContext.save()
                    finishedSession = session
                    showSummary = true
                }
            } else {
                let source: SessionSource = bleManager.isConnected ? .ble : .wifi
                sessionManager.startSession(source: source)

                let fmt = DateFormatter()
                fmt.locale = Locale(identifier: "zh_CN")
                fmt.dateFormat = "HH:mm"
                let title = "记录中 · \(fmt.string(from: Date()))"
                liveActivityManager.startActivity(title: title)
            }
        } label: {
            Image(systemName: sessionManager.isRecording ? "stop.circle.fill" : "record.circle")
                .font(.title3)
                .foregroundStyle(sessionManager.isRecording ? .red : (isLive ? .primary : .secondary))
                .symbolEffect(.pulse, isActive: sessionManager.isRecording)
        }
        .disabled(!isLive && !sessionManager.isRecording)
    }

    // MARK: - Start Focus

    // MARK: - Today Summary (重设计：紧凑横排 + 进度条)

    private var todaySummaryCard: some View {
        let totalMin = Int(todaySessions.reduce(0) { $0 + $1.duration } / 60)
        let avgVals = todaySessions.compactMap(\.avgStamina)
        let avgStamina = avgVals.isEmpty ? 0.0 : avgVals.reduce(0, +) / Double(avgVals.count)
        let unfeedbackCount = todaySessions.filter { $0.feedback == nil }.count

        return VStack(alignment: .leading, spacing: 12) {
            // 标题行
            HStack {
                Label("今日", systemImage: "chart.bar.fill")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                if unfeedbackCount > 0 {
                    Text("\(unfeedbackCount) 条待反馈")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
            }

            // 指标行：三个紧凑数据
            HStack(spacing: 0) {
                summaryMetric("\(todaySessions.count)", "场次", Color(.systemOrange))
                dividerLine
                summaryMetric(totalMin > 0 ? "\(totalMin)m" : "—", "时长", Color(.systemTeal))
                dividerLine
                summaryMetric(avgStamina > 0 ? "\(Int(avgStamina))" : "—", "续航", Color(.systemPink))
            }

            // 今日累计进度条（目标 4 小时）
            if totalMin > 0 {
                VStack(spacing: 4) {
                    ProgressView(value: min(Double(totalMin) / 240.0, 1.0))
                        .tint(Color(.systemTeal))
                    HStack {
                        Text("今日累计")
                            .font(.system(size: 9))
                            .foregroundStyle(.tertiary)
                        Spacer()
                        Text("\(totalMin) / 240 min")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.tertiary)
                    }
                }
            }
        }
        .padding(14)
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))
    }

    private func summaryMetric(_ value: String, _ label: String, _ tint: Color) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(size: 20, weight: .bold, design: .rounded))
                .foregroundStyle(tint)
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    private var dividerLine: some View {
        Rectangle()
            .fill(Color.primary.opacity(0.06))
            .frame(width: 1, height: 28)
    }

    @ViewBuilder
    private var recordingBar: some View {
        HStack(spacing: 12) {
            Circle().fill(.red).frame(width: 8, height: 8)
                .opacity(sessionManager.isPaused ? 0.3 : 1)

            Text(Flux.formatDuration(sessionManager.elapsed))
                .font(.system(size: 15, weight: .semibold, design: .monospaced))
                .contentTransition(.numericText())

            Spacer()

            if let seg = sessionManager.activeSegment {
                Text(seg.label.displayName)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(seg.label.color)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(seg.label.color.opacity(0.12), in: Capsule())
            }

            Button {
                showSegmentPicker = true
            } label: {
                Image(systemName: "rectangle.split.1x2")
                    .font(.subheadline)
            }
            .confirmationDialog("新建分段", isPresented: $showSegmentPicker) {
                ForEach(SegmentLabel.allCases) { label in
                    Button(label.displayName) { sessionManager.addSegment(label: label) }
                }
            }

            Button {
                sessionManager.isPaused ? sessionManager.resumeSession() : sessionManager.pauseSession()
            } label: {
                Image(systemName: sessionManager.isPaused ? "play.fill" : "pause.fill")
                    .font(.subheadline)
            }
        }
        .padding(12)
        .background(.red.opacity(0.06), in: .rect(cornerRadius: 16))
    }

    // MARK: - Daily Insight Card

    @ViewBuilder
    private var dailyInsightCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "brain.head.profile.fill")
                    .font(.caption)
                    .foregroundStyle(Flux.Colors.accent)
                Text("教练说")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                if isDailyInsightLoading {
                    ProgressView()
                        .controlSize(.mini)
                }
            }

            if let text = dailyInsightText {
                Text(text)
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .lineSpacing(4)
                    .fixedSize(horizontal: false, vertical: true)
            } else if !isDailyInsightLoading {
                Text("加载中…")
                    .font(.subheadline)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(14)
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: 16))
        .onAppear { loadDailyInsight() }
        .onChange(of: todaySessions.count) { _, _ in
            // session 数量变化时刷新
            loadDailyInsight()
        }
    }

    private func loadDailyInsight() {
        guard !isDailyInsightLoading else { return }
        isDailyInsightLoading = true

        Task {
            let text: String
            if #available(iOS 26.0, *) {
                text = await NLPSummaryEngine.shared.generateDailyInsight(sessions: todaySessions)
            } else {
                text = generateFallbackDailyInsight()
            }
            await MainActor.run {
                dailyInsightText = text
                isDailyInsightLoading = false
            }
        }
    }

    /// iOS 26 以下的 fallback（不依赖 NLPSummaryEngine）
    private func generateFallbackDailyInsight() -> String {
        let count = todaySessions.count
        let totalMin = Int(todaySessions.reduce(0) { $0 + $1.duration } / 60)
        let avgVals = todaySessions.compactMap(\.avgStamina)
        let avg = avgVals.isEmpty ? 0.0 : avgVals.reduce(0, +) / Double(avgVals.count)

        if avg >= 75 {
            return "今天状态不错，\(count) 段专注累计 \(totalMin) 分钟，续航稳定在 \(Int(avg))。保持这个节奏。"
        } else if avg >= 50 {
            return "今天 \(count) 段专注共 \(totalMin) 分钟，续航 \(Int(avg))，中规中矩。试试在疲劳前主动休息 5 分钟。"
        } else {
            return "今天身体信号偏弱，续航只有 \(Int(avg))，记得早点休息，明天会更好。"
        }
    }

    private func batteryIcon(_ level: Int) -> String {
        if level > 75 { return "battery.100" }
        if level > 50 { return "battery.75" }
        if level > 25 { return "battery.50" }
        return "battery.25"
    }
}

// MARK: - Dimensions Row (独立 struct，只在 value 变化时重绘)

private struct DimensionsRow: View, Equatable {
    let consistency: Double
    let tension: Double
    let fatigue: Double

    var body: some View {
        HStack(spacing: 12) {
            dimensionGauge("一致性", value: consistency, icon: "waveform.path", tint: Color(.systemTeal))
            dimensionGauge("紧张度", value: tension, icon: "arrow.up.right", tint: Color(.systemOrange).opacity(0.75))
            dimensionGauge("疲劳度", value: fatigue, icon: "flame", tint: Color(.systemPink).opacity(0.75))
        }
    }

    private func dimensionGauge(_ title: String, value: Double, icon: String, tint: Color) -> some View {
        VStack(spacing: 10) {
            Gauge(value: value) {
                Image(systemName: icon)
                    .font(.caption2)
            } currentValueLabel: {
                Text("\(Int(value * 100))")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
            }
            .gaugeStyle(.accessoryCircular)
            .tint(Gradient(colors: [tint.opacity(0.35), tint]))
            .scaleEffect(1.15)

            Text(title)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: 16))
    }
}

#Preview {
    DashboardView(showActiveSession: .constant(false), finishedSession: .constant(nil))
        .environmentObject(FluxService())
        .environmentObject(BLEManager())
        .environmentObject(SessionManager())
        .environmentObject(AlertManager())
        .environmentObject(LiveActivityManager())
}
