import SwiftUI
import SwiftData

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
                        .fill(Flux.Colors.accent.opacity(0.12))
                        .frame(width: 28, height: 28)
                    Image(systemName: "brain.head.profile.fill")
                        .font(.system(size: 13))
                        .foregroundStyle(Flux.Colors.accent)
                }
                Text("AI 教练")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                // NLP 引擎状态指示（长按可看详情）
                if #available(iOS 26.0, *) {
                    let nlpAvailable = NLPSummaryEngine.shared.isAvailable
                    Image(systemName: nlpAvailable ? "cpu.fill" : "cpu")
                        .font(.system(size: 9))
                        .foregroundStyle(nlpAvailable ? .green : .orange)
                        .help(NLPSummaryEngine.shared.diagnosticInfo)
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

            // 追问按钮行
            if dailyInsightText != nil {
                Divider()
                Button {
                    showCoachChat = true
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "bubble.left.and.text.bubble.right")
                            .font(.caption2)
                        Text("问教练")
                            .font(.caption.weight(.medium))
                    }
                    .foregroundStyle(Flux.Colors.accent)
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
                    text = NLPSummaryEngine.shared.generateEmptyDayInsight(recentSessions: recentSessions)
                } else {
                    text = await NLPSummaryEngine.shared.generateDailyInsight(sessions: todaySessions, allRecentSessions: recentSessions)
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
                                Text("常见问题")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)

                                ForEach(coachPresetQuestions, id: \.self) { q in
                                    Button {
                                        askCoach(question: q)
                                    } label: {
                                        Text(q)
                                            .font(.subheadline)
                                            .foregroundStyle(Flux.Colors.accent)
                                            .padding(.horizontal, 12)
                                            .padding(.vertical, 8)
                                            .background(Flux.Colors.accent.opacity(0.08), in: Capsule())
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
            .navigationTitle("问教练")
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

#Preview {
    DashboardView(showActiveSession: .constant(false), finishedSession: .constant(nil))
        .environmentObject(FluxService())
        .environmentObject(BLEManager())
        .environmentObject(SessionManager())
        .environmentObject(AlertManager())
        .environmentObject(LiveActivityManager())
        .modelContainer(for: [Session.self, Segment.self, FluxSnapshot.self, UserFeedback.self], inMemory: true)
}
