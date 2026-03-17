import SwiftUI
import SwiftData
import Charts

struct DashboardView: View {
    @EnvironmentObject var service: FluxService
    @EnvironmentObject var bleManager: BLEManager
    @EnvironmentObject var sessionManager: SessionManager
    @EnvironmentObject var alertManager: AlertManager
    @EnvironmentObject var liveActivityManager: LiveActivityManager
    @Environment(\.modelContext) private var modelContext

    @Query(sort: \Session.startedAt, order: .reverse) private var allSessions: [Session]

    @Binding var showActiveSession: Bool
    @Binding var finishedSession: Session?

    @State private var showSegmentPicker = false
    @State private var showFeedback = false
    @State private var showSummary = false   // recordButton 直接结束时用
    @State private var showConnectionSheet = false

    // Chart interaction
    @State private var selectedSessionDate: Date?
    @State private var selectedDayDate: Date?
    @State private var selectedTimeSlot: String?

    // Daily Insight
    @State private var dailyInsightText: String?
    @State private var isDailyInsightLoading = false

    // Coach Chat (B3/B4)
    @State private var showCoachChat = false
    @State private var coachMessages: [(question: String, answer: String)] = []
    @State private var isCoachLoading = false
    @State private var customQuestion = ""

    private let coachPresetQuestions: [String] = [
        "为什么我下午续航总是下降？",
        "怎样延长高效时间？",
        "我的紧张度正常吗？"
    ]

    /// 今日已完成的 Session（计算属性过滤，避免 init 中初始化 @Query 导致崩溃）
    private var todaySessions: [Session] {
        let startOfDay = Calendar.current.startOfDay(for: Date())
        return allSessions.filter { $0.startedAt >= startOfDay && $0.endedAt != nil }
    }

    /// 最近 7 天的 Session（用于趋势分析）
    private var recentSessions: [Session] {
        let weekAgo = Calendar.current.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        return allSessions.filter { $0.startedAt >= weekAgo && $0.endedAt != nil }
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

                    if isLive {
                        StaminaRingView(value: staminaValue, state: staminaState)
                            .drawingGroup()
                            .padding(.vertical, 4)
                    } else {
                        // 未连接：干净的占位
                        disconnectedPlaceholder
                    }

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
                    }

                    // 洞察组件区 (iOS Widget 风格)
                    insightWidgetGrid

                    dailyInsightCard
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
                updateWidgetData()
            }
            .onChange(of: todaySessions.count) { _, _ in
                updateWidgetData()
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
            .sheet(isPresented: $showConnectionSheet) {
                ConnectionGuideSheet()
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

    // MARK: - Connection Banner (紧凑 Capsule 风格)

    @ViewBuilder
    private var connectionBanner: some View {
        Button {
            showConnectionSheet = true
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "antenna.radiowaves.left.and.right.slash")
                    .font(.caption)
                    .symbolEffect(.pulse, options: .repeating)
                Text("未连接 · 点击连接")
                    .font(.caption.weight(.medium))
            }
            .foregroundStyle(.orange)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(.orange.opacity(0.1), in: Capsule())
        }
        .buttonStyle(.plain)
        .frame(maxWidth: .infinity, alignment: .center)
    }

    // MARK: - Disconnected Placeholder

    private var disconnectedPlaceholder: some View {
        VStack(spacing: 20) {
            ZStack {
                Circle()
                    .stroke(Color(.separator).opacity(0.2), lineWidth: 10)
                    .frame(width: 200, height: 200)

                VStack(spacing: 8) {
                    Image(systemName: "sensor.tag.radiowaves.forward")
                        .font(.system(size: 32))
                        .foregroundStyle(.quaternary)
                        .symbolEffect(.variableColor.iterative, options: .repeating)
                    Text("--")
                        .font(.system(size: 44, weight: .bold, design: .rounded))
                        .foregroundStyle(.quaternary)
                }
            }
            .padding(.vertical, 4)

            VStack(spacing: 8) {
                Text("等待设备连接")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)
                Button {
                    showConnectionSheet = true
                } label: {
                    Text("连接设备")
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 24)
                        .padding(.vertical, 10)
                        .background(Flux.Colors.accent, in: Capsule())
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)
            }
        }
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
            .background(.thinMaterial, in: .rect(cornerRadius: Flux.Radius.large))
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

            // Gauge 指标行
            HStack(spacing: 0) {
                summaryGauge(
                    value: Double(todaySessions.count) / 10.0,
                    label: "场次",
                    display: "\(todaySessions.count)",
                    tint: Color(.systemOrange)
                )
                dividerLine
                summaryGauge(
                    value: min(Double(totalMin) / 240.0, 1.0),
                    label: "时长",
                    display: totalMin > 0 ? "\(totalMin)m" : "—",
                    tint: Color(.systemTeal)
                )
                dividerLine
                summaryGauge(
                    value: avgStamina / 100.0,
                    label: "续航",
                    display: avgStamina > 0 ? "\(Int(avgStamina))" : "—",
                    tint: Flux.Colors.forStaminaValue(avgStamina)
                )
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
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: Flux.Radius.large))
    }

    private func summaryGauge(value: Double, label: String, display: String, tint: Color) -> some View {
        VStack(spacing: 6) {
            Gauge(value: min(max(value, 0), 1)) {
                EmptyView()
            } currentValueLabel: {
                Text(display)
                    .font(.system(size: 14, weight: .bold, design: .rounded))
            }
            .gaugeStyle(.accessoryCircular)
            .tint(Gradient(colors: [tint.opacity(0.3), tint]))
            .scaleEffect(0.9)

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

    // MARK: - Insight Widget Grid (iOS Widget 尺寸规范)

    private var insightWidgetGrid: some View {
        let spacing: CGFloat = 12

        return VStack(spacing: spacing) {
            // Row 1: 两个小组件 (square)
            HStack(spacing: spacing) {
                WidgetSessionCount(count: todaySessions.count, totalMin: Int(todaySessions.reduce(0) { $0 + $1.duration } / 60))
                WidgetBestSlot(sessions: todaySessions)
            }

            // Row 2: 中组件 — 今日续航趋势 (medium rectangle)
            if todaySessions.count >= 2 {
                TodayTrendChartCard(sessions: todaySessions, selectedDate: $selectedSessionDate)
            } else {
                WidgetPlaceholder(icon: "chart.xyaxis.line", title: "续航趋势", hint: "再完成 \(max(0, 2 - todaySessions.count)) 次记录解锁", style: .medium)
            }

            // Row 3: 中组件 — 7 天趋势 (medium rectangle)
            if recentSessions.count >= 2 {
                WeeklyTrendChartCard(sessions: recentSessions, selectedDate: $selectedDayDate)
            } else {
                WidgetPlaceholder(icon: "chart.bar.fill", title: "周趋势", hint: "再完成 \(max(0, 2 - recentSessions.count)) 次记录解锁", style: .medium)
            }

            // Row 4: 两个小组件 — 时段对比 + 平均续航
            HStack(spacing: spacing) {
                if todaySessions.count >= 2 {
                    TimeSlotChartCard(sessions: todaySessions, selectedSlot: $selectedTimeSlot)
                } else {
                    WidgetPlaceholder(icon: "clock.arrow.2.circlepath", title: "时段", hint: "多时段记录后解锁", style: .small)
                }
                WidgetAvgStamina(sessions: todaySessions)
            }
        }
    }

    // MARK: - Daily Insight Card

    @ViewBuilder
    private var dailyInsightCard: some View {
        let hasAnomalies: Bool = {
            if #available(iOS 26.0, *) {
                return !NLPSummaryEngine.shared.detectDailyAnomalies(sessions: todaySessions).isEmpty
            }
            return false
        }()
        let hasCriticalAnomaly: Bool = {
            if #available(iOS 26.0, *) {
                return NLPSummaryEngine.shared.detectDailyAnomalies(sessions: todaySessions).contains { $0.severity == .critical }
            }
            return false
        }()
        let anomalyCount: Int = {
            if #available(iOS 26.0, *) {
                return NLPSummaryEngine.shared.detectDailyAnomalies(sessions: todaySessions).count
            }
            return 0
        }()

        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                ZStack {
                    Circle()
                        .fill(Color.cyan.opacity(0.12))
                        .frame(width: 28, height: 28)
                    Image(systemName: "waveform.path")
                        .font(.system(size: 13))
                        .foregroundStyle(.cyan)
                }
                Text("体能洞察")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                // 引擎状态指示
                if #available(iOS 26.0, *) {
                    let engineAvailable = BodyInsightEngine.shared.isAvailable
                    Image(systemName: engineAvailable ? "cpu.fill" : "cpu")
                        .font(.system(size: 9))
                        .foregroundStyle(engineAvailable ? .green : .orange)
                        .help(BodyInsightEngine.shared.diagnosticInfo)
                }

                // 异常 badge
                if hasAnomalies {
                    HStack(spacing: 3) {
                        Image(systemName: hasCriticalAnomaly ? "exclamationmark.triangle.fill" : "info.circle.fill")
                            .font(.system(size: 9))
                        Text("\(anomalyCount) 项异常")
                            .font(.system(size: 10, weight: .medium))
                    }
                    .foregroundStyle(hasCriticalAnomaly ? .red : .orange)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background((hasCriticalAnomaly ? Color.red : Color.orange).opacity(0.1), in: Capsule())
                }

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

            // 追问按钮行（可选：了解自己的状态数据）
            if dailyInsightText != nil {
                Divider()
                Button {
                    showCoachChat = true
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "info.circle")
                            .font(.caption2)
                        Text("了解数据含义")
                            .font(.caption.weight(.medium))
                    }
                    .foregroundStyle(.cyan)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(14)
        .background(
            LinearGradient(
                colors: [Flux.Colors.accent.opacity(0.06), Color(.secondarySystemGroupedBackground)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: .rect(cornerRadius: Flux.Radius.large)
        )
        .onAppear { loadDailyInsight() }
        .onChange(of: todaySessions.count) { _, _ in
            loadDailyInsight()
        }
        .sheet(isPresented: $showCoachChat) {
            coachChatSheet
        }
    }

    private func loadDailyInsight() {
        guard !isDailyInsightLoading else { return }
        isDailyInsightLoading = true

        Task {
            let text: String
            if #available(iOS 26.0, *) {
                if todaySessions.isEmpty {
                    text = "今天还没有记录。连上设备开始专注，让身体数据帮你了解自己的状态。"
                } else {
                    text = await BodyInsightEngine.shared.generateDailySummary(sessions: todaySessions)
                }
            } else {
                text = generateFallbackDailyInsight()
            }
            await MainActor.run {
                dailyInsightText = text
                isDailyInsightLoading = false
            }
        }
    }

    // MARK: - Coach Chat Sheet (B3/B4)

    private var coachChatSheet: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        // 预设问题
                        if coachMessages.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("想知道数据背后的含义？")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)

                                ForEach(coachPresetQuestions, id: \.self) { q in
                                    Button {
                                        askCoach(question: q)
                                    } label: {
                                        Text(q)
                                            .font(.subheadline)
                                            .foregroundStyle(.cyan)
                                            .padding(.horizontal, 12)
                                            .padding(.vertical, 8)
                                            .background(Color.cyan.opacity(0.08), in: Capsule())
                                    }
                                    .buttonStyle(.plain)
                                    .disabled(isCoachLoading)
                                }
                            }
                            .padding(.horizontal)
                        }

                        // 消息列表
                        ForEach(Array(coachMessages.enumerated()), id: \.offset) { idx, msg in
                            VStack(alignment: .leading, spacing: 8) {
                                // 用户问题
                                HStack {
                                    Spacer()
                                    Text(msg.question)
                                        .font(.subheadline)
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 8)
                                        .background(Flux.Colors.accent.opacity(0.12), in: .rect(cornerRadius: 12))
                                }

                                // 教练回答
                                HStack(alignment: .top, spacing: 8) {
                                    Image(systemName: "brain.head.profile.fill")
                                        .font(.caption)
                                        .foregroundStyle(Flux.Colors.accent)
                                        .padding(.top, 4)
                                    Text(msg.answer)
                                        .font(.subheadline)
                                        .lineSpacing(3)
                                }
                            }
                            .padding(.horizontal)
                            .id(idx)
                        }

                        if isCoachLoading {
                            HStack(spacing: 8) {
                                Image(systemName: "brain.head.profile.fill")
                                    .font(.caption)
                                    .foregroundStyle(Flux.Colors.accent)
                                ProgressView()
                                    .controlSize(.small)
                                Text("思考中…")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.horizontal)
                        }
                    }
                    .padding(.vertical)
                }
                .onChange(of: coachMessages.count) { _, _ in
                    withAnimation {
                        proxy.scrollTo(coachMessages.count - 1, anchor: .bottom)
                    }
                }
            }
            .safeAreaInset(edge: .bottom) {
                HStack(spacing: 8) {
                    TextField("输入你的问题…", text: $customQuestion)
                        .textFieldStyle(.roundedBorder)
                        .font(.subheadline)
                        .submitLabel(.send)
                        .onSubmit { sendCustomQuestion() }

                    Button {
                        sendCustomQuestion()
                    } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.title2)
                            .foregroundStyle(customQuestion.isEmpty ? .gray : Flux.Colors.accent)
                    }
                    .disabled(customQuestion.isEmpty || isCoachLoading)
                    .buttonStyle(.plain)
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(.bar)
            }
            .navigationTitle("了解数据")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { showCoachChat = false }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private func sendCustomQuestion() {
        let q = customQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return }
        customQuestion = ""
        askCoach(question: q)
    }

    private func askCoach(question: String) {
        guard !isCoachLoading else { return }
        isCoachLoading = true

        Task {
            let answer: String
            if #available(iOS 26.0, *) {
                let anomalies = NLPSummaryEngine.shared.detectDailyAnomalies(sessions: todaySessions)
                let weeklyTrend = recentSessions.count >= 2
                    ? NLPSummaryEngine.shared.generateWeeklyTrend(sessions: recentSessions)
                    : nil

                let context = NLPSummaryEngine.CoachContext(
                    todaySessions: todaySessions,
                    dailyInsight: dailyInsightText,
                    anomalies: anomalies,
                    weeklyTrend: weeklyTrend
                )

                answer = await NLPSummaryEngine.shared.askFollowUp(context: context, question: question)
            } else {
                answer = "AI 教练需要 iOS 26.0 或更高版本。建议每 25 分钟主动休息 5 分钟，保持良好的专注节奏。"
            }

            await MainActor.run {
                coachMessages.append((question: question, answer: answer))
                isCoachLoading = false
            }
        }
    }

    /// iOS 26 以下的 fallback（不依赖 NLPSummaryEngine）
    private func generateFallbackDailyInsight() -> String {
        guard !todaySessions.isEmpty else {
            let hour = Calendar.current.component(.hour, from: Date())
            if hour < 12 {
                return "早上好！连上设备开始今天的第一段专注吧。"
            } else if hour < 18 {
                return "下午好，今天还没有专注记录。找个时间段开始一段吧。"
            } else {
                return "今天还没有记录，没关系，适当休息也是提升的一部分。"
            }
        }

        let count = todaySessions.count
        let totalMin = Int(todaySessions.reduce(0) { $0 + $1.duration } / 60)
        let avgVals = todaySessions.compactMap(\.avgStamina)
        let avg = avgVals.isEmpty ? 0.0 : avgVals.reduce(0, +) / Double(avgVals.count)

        if avg >= 75 {
            return [
                "今天 \(count) 段专注累计 \(totalMin) 分钟，续航 \(Int(avg))，状态很好。",
                "身体信号稳定，\(totalMin) 分钟专注、续航 \(Int(avg))，保持这个节奏。"
            ].randomElement()!
        } else if avg >= 50 {
            return [
                "今天 \(count) 段专注共 \(totalMin) 分钟，续航 \(Int(avg))，试试在疲劳前主动休息 5 分钟。",
                "\(totalMin) 分钟专注，续航 \(Int(avg))，有提升空间——关键在休息节奏。"
            ].randomElement()!
        } else {
            return [
                "今天身体信号偏弱，续航 \(Int(avg))，早点休息明天会更好。",
                "续航 \(Int(avg)) 偏低，身体需要恢复，不要勉强自己。"
            ].randomElement()!
        }
    }

    private func batteryIcon(_ level: Int) -> String {
        if level > 75 { return "battery.100" }
        if level > 50 { return "battery.75" }
        if level > 25 { return "battery.50" }
        return "battery.25"
    }

    /// 向 Widget Extension 写入最新数据
    private func updateWidgetData() {
        WidgetDataManager.updateFromSessions(
            today: todaySessions,
            recent: recentSessions
        )
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
        .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.large))
    }
}

// MARK: - iOS Widget Design Constants

private enum WidgetStyle {
    /// iOS Widget 标准圆角 (iPhone 15 系列)
    static let cornerRadius: CGFloat = 22
    /// 内边距
    static let padding: CGFloat = 16
    /// 小组件最小高度 (正方形，宽度由 grid 决定)
    static let smallHeight: CGFloat = 170
    /// 中组件高度 (宽矩形)
    static let mediumHeight: CGFloat = 170
}

// MARK: - Widget Placeholder (占位组件)

private struct WidgetPlaceholder: View {
    let icon: String
    let title: String
    let hint: String
    let style: WidgetSize

    enum WidgetSize { case small, medium }

    var body: some View {
        VStack(spacing: 8) {
            Spacer()
            Image(systemName: icon)
                .font(.system(size: style == .small ? 24 : 28))
                .foregroundStyle(.quaternary)
            Text(title)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.tertiary)
            Text(hint)
                .font(.system(size: 11))
                .foregroundStyle(.quaternary)
                .multilineTextAlignment(.center)
                .lineLimit(2)
            Spacer()
        }
        .padding(WidgetStyle.padding)
        .frame(maxWidth: .infinity)
        .frame(height: style == .small ? WidgetStyle.smallHeight : WidgetStyle.mediumHeight)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: WidgetStyle.cornerRadius))
    }
}

// MARK: - Small Widget: 今日场次 (正方形)

private struct WidgetSessionCount: View {
    let count: Int
    let totalMin: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Image(systemName: "flame.fill")
                    .font(.system(size: 13))
                    .foregroundStyle(.orange)
                Spacer()
            }

            Spacer()

            Text("\(count)")
                .font(.system(size: 44, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
                .contentTransition(.numericText())

            Text("场专注")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)

            Spacer().frame(height: 6)

            Text(totalMin > 0 ? "累计 \(totalMin) 分钟" : "今日未记录")
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
        }
        .padding(WidgetStyle.padding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(height: WidgetStyle.smallHeight)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: WidgetStyle.cornerRadius))
    }
}

// MARK: - Small Widget: 最佳时段 (正方形)

private struct WidgetBestSlot: View {
    let sessions: [Session]

    private var bestSlot: (name: String, avg: Double, icon: String)? {
        var map: [String: [Double]] = [:]
        for s in sessions {
            guard let avg = s.avgStamina else { continue }
            let hour = Calendar.current.component(.hour, from: s.startedAt)
            let slot: String
            switch hour {
            case 6..<12:  slot = "上午"
            case 12..<14: slot = "午间"
            case 14..<18: slot = "下午"
            case 18..<22: slot = "晚间"
            default:      slot = "其他"
            }
            map[slot, default: []].append(avg)
        }
        guard let best = map.max(by: {
            $0.value.reduce(0, +) / Double($0.value.count) <
            $1.value.reduce(0, +) / Double($1.value.count)
        }) else { return nil }

        let avg = best.value.reduce(0, +) / Double(best.value.count)
        let icon: String
        switch best.key {
        case "上午": icon = "sunrise.fill"
        case "午间": icon = "sun.max.fill"
        case "下午": icon = "sun.haze.fill"
        case "晚间": icon = "moon.stars.fill"
        default:     icon = "clock.fill"
        }
        return (best.key, avg, icon)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Image(systemName: "clock.arrow.circlepath")
                    .font(.system(size: 13))
                    .foregroundStyle(.cyan)
                Spacer()
            }

            Spacer()

            if let slot = bestSlot {
                Image(systemName: slot.icon)
                    .font(.system(size: 22))
                    .foregroundStyle(Flux.Colors.forStaminaValue(slot.avg))
                    .padding(.bottom, 4)

                Text(slot.name)
                    .font(.system(size: 22, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)

                Text("最佳时段")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)

                Spacer().frame(height: 6)

                Text("平均续航 \(Int(slot.avg))")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
            } else {
                Text("--")
                    .font(.system(size: 44, weight: .bold, design: .rounded))
                    .foregroundStyle(.quaternary)
                Text("最佳时段")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.tertiary)

                Spacer().frame(height: 6)

                Text("记录后解锁")
                    .font(.system(size: 11))
                    .foregroundStyle(.quaternary)
            }
        }
        .padding(WidgetStyle.padding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(height: WidgetStyle.smallHeight)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: WidgetStyle.cornerRadius))
    }
}

// MARK: - Small Widget: 平均续航 (正方形)

private struct WidgetAvgStamina: View {
    let sessions: [Session]

    private var avgStamina: Double {
        let vals = sessions.compactMap(\.avgStamina)
        guard !vals.isEmpty else { return 0 }
        return vals.reduce(0, +) / Double(vals.count)
    }

    var body: some View {
        let color = Flux.Colors.forStaminaValue(avgStamina)

        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Image(systemName: "bolt.fill")
                    .font(.system(size: 13))
                    .foregroundStyle(color)
                Spacer()
            }

            Spacer()

            if avgStamina > 0 {
                Text("\(Int(avgStamina))")
                    .font(.system(size: 44, weight: .bold, design: .rounded))
                    .foregroundStyle(color)
                    .contentTransition(.numericText())

                Text("平均续航")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)

                Spacer().frame(height: 6)

                // 迷你进度条
                ProgressView(value: avgStamina / 100)
                    .tint(color)
            } else {
                Text("--")
                    .font(.system(size: 44, weight: .bold, design: .rounded))
                    .foregroundStyle(.quaternary)

                Text("平均续航")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.tertiary)

                Spacer().frame(height: 6)

                Text("开始记录后显示")
                    .font(.system(size: 11))
                    .foregroundStyle(.quaternary)
            }
        }
        .padding(WidgetStyle.padding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(height: WidgetStyle.smallHeight)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: WidgetStyle.cornerRadius))
    }
}

// MARK: - Medium Widget: 今日续航趋势 (宽矩形)

private struct TodayTrendChartCard: View {
    let sessions: [Session]
    @Binding var selectedDate: Date?

    private var sortedSessions: [Session] {
        sessions.sorted { $0.startedAt < $1.startedAt }
    }

    private var selected: Session? {
        guard let date = selectedDate else { return nil }
        return sortedSessions.min(by: {
            abs($0.startedAt.timeIntervalSince(date)) < abs($1.startedAt.timeIntervalSince(date))
        })
    }

    private var displayStamina: Double {
        selected?.avgStamina ?? sortedSessions.last?.avgStamina ?? 0
    }

    var body: some View {
        let color = Flux.Colors.forStaminaValue(displayStamina)

        VStack(alignment: .leading, spacing: 8) {
            headerRow(color: color)
            chartView(color: color)
        }
        .padding(WidgetStyle.padding)
        .frame(height: WidgetStyle.mediumHeight)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: WidgetStyle.cornerRadius))
        .animation(.snappy, value: selectedDate)
    }

    private func headerRow(color: Color) -> some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text("续航趋势")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                HStack(alignment: .firstTextBaseline, spacing: 3) {
                    Text("\(Int(displayStamina))")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundStyle(color)
                        .contentTransition(.numericText(value: displayStamina))
                        .animation(.snappy(duration: 0.3), value: Int(displayStamina))
                    Text("avg")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
            }
            Spacer()
            if let s = selected {
                Text(s.startedAt, format: .dateTime.hour().minute())
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .transition(.opacity)
            }
        }
    }

    private func chartView(color: Color) -> some View {
        let selectedID = selected?.id

        return Chart(sortedSessions, id: \.id) { session in
            let avg: Double = session.avgStamina ?? 0
            let isSelected: Bool = selectedID == session.id

            LineMark(
                x: .value("时间", session.startedAt),
                y: .value("续航", avg)
            )
            .interpolationMethod(.catmullRom)
            .foregroundStyle(Flux.Colors.accent.gradient)
            .lineStyle(StrokeStyle(lineWidth: 2))

            AreaMark(
                x: .value("时间", session.startedAt),
                yStart: .value("底", 0),
                yEnd: .value("续航", avg)
            )
            .interpolationMethod(.catmullRom)
            .foregroundStyle(
                .linearGradient(
                    colors: [Flux.Colors.accent.opacity(0.2), .clear],
                    startPoint: .top, endPoint: .bottom
                )
            )

            PointMark(
                x: .value("时间", session.startedAt),
                y: .value("续航", avg)
            )
            .symbolSize(isSelected ? 60 : 20)
            .foregroundStyle(isSelected ? color : Flux.Colors.accent)
        }
        .chartYScale(domain: 0...100)
        .chartXSelection(value: $selectedDate)
        .chartYAxis(.hidden)
        .chartXAxis(.hidden)
    }
}

// MARK: - Medium Widget: 7 天趋势 (宽矩形)

private struct DayData: Identifiable {
    let id: Date
    let date: Date
    let avgStamina: Double
    let sessionCount: Int
    let totalMin: Int
}

private struct WeeklyTrendChartCard: View {
    let sessions: [Session]
    @Binding var selectedDate: Date?

    private var days: [DayData] {
        let cal = Calendar.current
        let grouped = Dictionary(grouping: sessions) { cal.startOfDay(for: $0.startedAt) }
        return grouped.map { (date, daySessions) in
            let staminas = daySessions.compactMap(\.avgStamina)
            let avg = staminas.isEmpty ? 0 : staminas.reduce(0, +) / Double(staminas.count)
            let totalMin = Int(daySessions.reduce(0) { $0 + $1.duration } / 60)
            return DayData(id: date, date: date, avgStamina: avg, sessionCount: daySessions.count, totalMin: totalMin)
        }.sorted { $0.date < $1.date }
    }

    private var selected: DayData? {
        guard let date = selectedDate else { return nil }
        return days.min(by: {
            abs($0.date.timeIntervalSince(date)) < abs($1.date.timeIntervalSince(date))
        })
    }

    private var displayStamina: Double {
        selected?.avgStamina ?? days.last?.avgStamina ?? 0
    }

    var body: some View {
        let color = Flux.Colors.forStaminaValue(displayStamina)

        VStack(alignment: .leading, spacing: 8) {
            headerRow(color: color)
            chartView(color: color)
        }
        .padding(WidgetStyle.padding)
        .frame(height: WidgetStyle.mediumHeight)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: WidgetStyle.cornerRadius))
        .animation(.snappy, value: selectedDate)
    }

    private func headerRow(color: Color) -> some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text("近 7 天")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                HStack(alignment: .firstTextBaseline, spacing: 3) {
                    Text("\(Int(displayStamina))")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundStyle(color)
                        .contentTransition(.numericText(value: displayStamina))
                        .animation(.snappy(duration: 0.3), value: Int(displayStamina))
                    Text("avg")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
            }
            Spacer()
            if let d = selected {
                VStack(alignment: .trailing, spacing: 1) {
                    Text(d.date, format: .dateTime.month().day())
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                    Text("\(d.sessionCount)次 \(d.totalMin)m")
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
                }
                .transition(.opacity)
            }
        }
    }

    private func chartView(color: Color) -> some View {
        let selectedDay = selected?.date

        return Chart(days) { day in
            let isSelected: Bool = selectedDay == day.date
            let barColor: AnyShapeStyle = isSelected
                ? AnyShapeStyle(color.gradient)
                : AnyShapeStyle(Flux.Colors.accent.opacity(0.5).gradient)

            BarMark(
                x: .value("日期", day.date, unit: .day),
                y: .value("续航", day.avgStamina)
            )
            .foregroundStyle(barColor)
            .cornerRadius(4)
        }
        .chartYScale(domain: 0...100)
        .chartXSelection(value: $selectedDate)
        .chartYAxis(.hidden)
        .chartXAxis {
            AxisMarks(values: .stride(by: .day)) { _ in
                AxisValueLabel(format: .dateTime.weekday(.narrow))
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
            }
        }
    }
}

// MARK: - Small Widget: 时段对比 (正方形 mini bar)

private struct SlotData: Identifiable {
    var id: String { slot }
    let slot: String
    let avgStamina: Double
    let count: Int
    let order: Int
}

private struct TimeSlotChartCard: View {
    let sessions: [Session]
    @Binding var selectedSlot: String?

    private static let slotOrder = ["上午": 0, "午间": 1, "下午": 2, "晚间": 3]

    private var slots: [SlotData] {
        var map: [String: [Double]] = [:]
        for s in sessions {
            guard let avg = s.avgStamina else { continue }
            let hour = Calendar.current.component(.hour, from: s.startedAt)
            let slot: String
            switch hour {
            case 6..<12:  slot = "上午"
            case 12..<14: slot = "午间"
            case 14..<18: slot = "下午"
            case 18..<22: slot = "晚间"
            default:      slot = "其他"
            }
            map[slot, default: []].append(avg)
        }
        return map.map { (slot, values) in
            SlotData(
                slot: slot,
                avgStamina: values.reduce(0, +) / Double(values.count),
                count: values.count,
                order: Self.slotOrder[slot] ?? 4
            )
        }.sorted { $0.order < $1.order }
    }

    private var selectedData: SlotData? {
        guard let name = selectedSlot else { return nil }
        return slots.first { $0.slot == name }
    }

    private var displayStamina: Double {
        selectedData?.avgStamina ?? slots.max(by: { $0.avgStamina < $1.avgStamina })?.avgStamina ?? 0
    }

    var body: some View {
        let color = Flux.Colors.forStaminaValue(displayStamina)

        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Image(systemName: "chart.bar.fill")
                    .font(.system(size: 13))
                    .foregroundStyle(.purple)
                Spacer()
            }

            Spacer().frame(height: 8)

            Text("\(Int(displayStamina))")
                .font(.system(size: 28, weight: .bold, design: .rounded))
                .foregroundStyle(color)
                .contentTransition(.numericText(value: displayStamina))
                .animation(.snappy(duration: 0.3), value: Int(displayStamina))

            Text(selectedData?.slot ?? "时段")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)

            Spacer()

            chartView
        }
        .padding(WidgetStyle.padding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(height: WidgetStyle.smallHeight)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: WidgetStyle.cornerRadius))
        .animation(.snappy, value: selectedSlot)
    }

    private var chartView: some View {
        Chart(slots) { slot in
            let isSelected: Bool = selectedSlot == slot.slot
            let barColor: AnyShapeStyle = isSelected
                ? AnyShapeStyle(Flux.Colors.forStaminaValue(slot.avgStamina).gradient)
                : AnyShapeStyle(Flux.Colors.accent.opacity(0.4).gradient)

            BarMark(
                x: .value("时段", slot.slot),
                y: .value("续航", slot.avgStamina)
            )
            .foregroundStyle(barColor)
            .cornerRadius(3)
        }
        .chartYScale(domain: 0...100)
        .chartXSelection(value: $selectedSlot)
        .chartYAxis(.hidden)
        .chartXAxis {
            AxisMarks { _ in
                AxisValueLabel()
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(height: 44)
    }
}

#Preview {
    DashboardView(showActiveSession: .constant(false), finishedSession: .constant(nil))
        .environmentObject(FluxService())
        .environmentObject(BLEManager())
        .environmentObject(SessionManager())
        .environmentObject(AlertManager())
        .environmentObject(LiveActivityManager())
        .modelContainer(for: [Session.self, Segment.self, FluxSnapshot.self, UserFeedback.self], inMemory: true)
}
