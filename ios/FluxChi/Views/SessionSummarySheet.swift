import SwiftUI
import SwiftData

struct SessionSummarySheet: View {
    let session: Session
    @Environment(\.modelContext) private var modelContext
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var personalization: PersonalizationManager

    @State private var aiSummary: String?
    @State private var isLoadingAI = false
    @State private var selectedMood: Mood?
    @State private var showCorrection = false
    @State private var accuracyRating: Int = 3
    @State private var appeared = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollView {
                    VStack(spacing: Flux.Spacing.section) {
                        headerSection
                        statsCards
                        quickFeedbackSection
                    }
                    .padding()
                    .padding(.bottom, 20)
                }
                .scrollBounceBehavior(.basedOnSize)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("专注完成")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("完成") { dismiss() }
                        .fontWeight(.semibold)
                }
            }
        }
        .onAppear {
            guard !appeared else { return }
            appeared = true
            loadAISummary()
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(spacing: Flux.Spacing.group) {
            ZStack {
                // 撒花粒子动画（Canvas 渲染，不影响布局）
                ConfettiCanvas()
                    .frame(height: 120)
                    .allowsHitTesting(false)

                VStack(spacing: 8) {
                    Image(systemName: achievementIcon)
                        .font(.system(size: 44))
                        .foregroundStyle(achievementColor.gradient)

                    Text(achievementTitle)
                        .font(.title2.bold())
                        .foregroundStyle(.primary)

                    Text(Flux.formatDurationLong(session.duration))
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 8)

            aiSummaryCard
        }
    }

    private var aiSummaryCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .font(.caption)
                    .foregroundStyle(.purple)
                Text("智能总结")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.purple)

                if isLoadingAI {
                    ProgressView()
                        .scaleEffect(0.6)
                }
            }

            if let summary = aiSummary {
                Text(summary)
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .lineSpacing(4)
                    .fixedSize(horizontal: false, vertical: true)
            } else if !isLoadingAI {
                Text(session.summaryText ?? fallbackText)
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .lineSpacing(4)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.7)
                    Text("正在生成…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(height: 40)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: Flux.Radius.large))
    }

    // MARK: - Stats Cards

    private var statsCards: some View {
        VStack(spacing: Flux.Spacing.item) {
            HStack(spacing: Flux.Spacing.item) {
                statCard(
                    title: "总耗时",
                    value: Flux.formatDuration(session.duration),
                    icon: "clock.fill",
                    tint: Color(.systemTeal)
                )
                statCard(
                    title: "平均续航",
                    value: session.avgStamina.map { "\(Int($0))" } ?? "—",
                    icon: "heart.fill",
                    tint: Color(.systemPink).opacity(0.8)
                )
            }

            HStack(spacing: Flux.Spacing.item) {
                statCard(
                    title: "最低续航",
                    value: session.minStamina.map { "\(Int($0))" } ?? "—",
                    icon: "arrow.down.circle.fill",
                    tint: Color(.systemOrange).opacity(0.8)
                )
                statCard(
                    title: "分段",
                    value: "\(session.segments.count)",
                    icon: "rectangle.split.1x2.fill",
                    tint: Color(.systemGreen).opacity(0.8)
                )
            }
        }
    }

    private func statCard(title: String, value: String, icon: String, tint: Color) -> some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(tint)

            Text(value)
                .font(.system(size: 24, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(14)
        .frame(maxWidth: .infinity)
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: Flux.Radius.medium))
    }

    // MARK: - Quick Feedback + Correction

    private var quickFeedbackSection: some View {
        VStack(spacing: Flux.Spacing.item) {
            // 心情打分
            Text("这次感觉怎么样？")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.primary)

            HStack(spacing: 16) {
                ForEach(Mood.allCases) { mood in
                    Button {
                        withAnimation(.spring(duration: 0.3)) {
                            selectedMood = mood
                        }
                        saveMoodFeedback(mood)
                    } label: {
                        VStack(spacing: 6) {
                            Text(mood.emoji)
                                .font(.system(size: 36))
                                .scaleEffect(selectedMood == mood ? 1.15 : 1.0)

                            Text(mood.label)
                                .font(.caption2)
                                .foregroundStyle(selectedMood == mood ? .primary : .secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(
                            selectedMood == mood
                                ? AnyShapeStyle(mood.color.opacity(0.12))
                                : AnyShapeStyle(Color.clear),
                            in: .rect(cornerRadius: Flux.Radius.medium)
                        )
                    }
                    .buttonStyle(.plain)
                    .sensoryFeedback(.selection, trigger: selectedMood)
                }
            }

            if selectedMood != nil {
                // 准确度纠正入口
                Divider()
                    .padding(.vertical, 4)

                if !showCorrection {
                    Button {
                        withAnimation(.spring(duration: 0.3)) {
                            showCorrection = true
                        }
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "slider.horizontal.3")
                                .font(.caption)
                            Text("结果不准？帮我纠正")
                                .font(.caption)
                        }
                        .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                } else {
                    correctionSection
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground), in: .rect(cornerRadius: Flux.Radius.large))
    }

    // MARK: - Correction Section

    private var correctionSection: some View {
        VStack(alignment: .leading, spacing: Flux.Spacing.item) {
            Text("系统判断准确吗？")
                .font(.caption.weight(.medium))
                .foregroundStyle(.primary)

            HStack(spacing: 8) {
                ForEach(1...5, id: \.self) { i in
                    Button {
                        withAnimation(.snappy) { accuracyRating = i }
                        saveAccuracyRating(i)
                    } label: {
                        Image(systemName: i <= accuracyRating ? "star.fill" : "star")
                            .font(.title3)
                            .foregroundStyle(i <= accuracyRating ? Color(.systemYellow) : .secondary)
                    }
                    .buttonStyle(.plain)
                }

                Spacer()

                Text(accuracyLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // 系统预测 vs 实际感受对比
            if let avg = session.avgStamina, let mood = selectedMood {
                let predicted = staminaStateFor(avg)
                HStack {
                    VStack(spacing: 4) {
                        Text("系统预测")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        FluxStatusBadge(
                            label: predicted.displayName,
                            icon: predicted.systemImage,
                            tint: Flux.Colors.forStaminaState(predicted),
                            isActive: false
                        )
                    }
                    .frame(maxWidth: .infinity)

                    Image(systemName: "arrow.left.arrow.right")
                        .font(.caption2)
                        .foregroundStyle(.quaternary)

                    VStack(spacing: 4) {
                        Text("实际感受")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        FluxStatusBadge(
                            label: mood.label,
                            icon: moodSystemIcon(mood),
                            tint: mood.color,
                            isActive: false
                        )
                    }
                    .frame(maxWidth: .infinity)
                }
                .padding(10)
                .background(Color(.tertiarySystemGroupedBackground), in: .rect(cornerRadius: Flux.Radius.medium))
            }

            HStack(spacing: 4) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.green)
                Text("反馈已保存，模型会持续学习你的特征")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Achievement Logic

    private var achievementIcon: String {
        guard let avg = session.avgStamina else { return "checkmark.circle.fill" }
        if avg >= 80 { return "star.fill" }
        if avg >= 60 { return "flame.fill" }
        if avg >= 40 { return "checkmark.circle.fill" }
        return "leaf.fill"
    }

    private var achievementTitle: String {
        guard let avg = session.avgStamina else { return "专注完成" }
        if avg >= 80 { return "出色表现" }
        if avg >= 60 { return "不错的专注" }
        if avg >= 40 { return "稳步前进" }
        return "坚持就是胜利"
    }

    private var achievementColor: Color {
        guard let avg = session.avgStamina else { return Color(.systemGreen) }
        if avg >= 80 { return Color(.systemOrange) }   // 亮色下 yellow 不可见，用 orange
        if avg >= 60 { return Color(.systemOrange).opacity(0.8) }
        if avg >= 40 { return Color(.systemGreen) }
        return Color(.systemTeal)
    }

    private var accuracyLabel: String {
        switch accuracyRating {
        case 1:  return "完全不准"
        case 2:  return "不太准"
        case 3:  return "一般"
        case 4:  return "比较准"
        case 5:  return "非常准"
        default: return ""
        }
    }

    private var fallbackText: String {
        let min = Int(session.duration / 60)
        return "本次专注共 \(min) 分钟，共 \(session.segments.count) 个分段。"
    }

    // MARK: - Helpers

    private func staminaStateFor(_ value: Double) -> StaminaState {
        if value > 60 { return .focused }
        if value > 30 { return .fading }
        return .depleted
    }

    private func moodSystemIcon(_ mood: Mood) -> String {
        switch mood {
        case .great:     return "face.smiling.inverse"
        case .okay:      return "face.smiling"
        case .exhausted: return "zzz"
        }
    }

    // MARK: - Actions

    private func loadAISummary() {
        guard #available(iOS 26.0, *) else { return }
        isLoadingAI = true
        Task {
            let summary = await NLPSummaryEngine.shared.generateSummary(for: session)
            withAnimation {
                aiSummary = summary
                isLoadingAI = false
            }
        }
    }

    private func saveMoodFeedback(_ mood: Mood) {
        let feeling: UserFeeling
        switch mood {
        case .great:     feeling = .focused
        case .okay:      feeling = .okay
        case .exhausted: feeling = .exhausted
        }

        if session.feedback == nil {
            let fb = UserFeedback(feeling: feeling, accuracyRating: accuracyRating, notes: "")
            fb.session = session
            session.feedback = fb
            modelContext.insert(fb)
        } else {
            session.feedback?.feeling = feeling
        }
        try? modelContext.save()

        if let fb = session.feedback {
            personalization.addTrainingData(session: session, feedback: fb)
        }
    }

    private func saveAccuracyRating(_ rating: Int) {
        guard let fb = session.feedback else { return }
        fb.accuracyRating = rating
        try? modelContext.save()

        // 低准确度 (1-2) 时，额外触发一次学习以加大纠正力度
        if rating <= 2 {
            personalization.addTrainingData(session: session, feedback: fb)
        }
    }
}

// MARK: - Mood Enum

enum Mood: String, CaseIterable, Identifiable {
    case great, okay, exhausted

    var id: String { rawValue }

    var emoji: String {
        switch self {
        case .great:     return "🤩"
        case .okay:      return "🙂"
        case .exhausted: return "😫"
        }
    }

    var label: String {
        switch self {
        case .great:     return "很棒"
        case .okay:      return "还行"
        case .exhausted: return "累了"
        }
    }

    var color: Color {
        switch self {
        case .great:     return Color(.systemGreen)
        case .okay:      return Color(.systemTeal)
        case .exhausted: return Color(.systemOrange)
        }
    }
}

// MARK: - Confetti Canvas (GPU 渲染，不触发布局重计算)

private struct ConfettiCanvas: View {
    @State private var particles: [ConfettiParticle] = []
    @State private var startTime: Date = .now

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30)) { timeline in
            Canvas { context, size in
                let elapsed = timeline.date.timeIntervalSince(startTime)
                for p in particles {
                    let age = elapsed - p.delay
                    guard age > 0, age < p.lifetime else { continue }

                    let progress = age / p.lifetime
                    let x = p.startX * size.width + sin(age * p.wobble) * 15
                    let y = p.startY * size.height + age * p.speed
                    let opacity = 1.0 - progress

                    let rect = CGRect(
                        x: x - p.size / 2,
                        y: y - p.size / 2,
                        width: p.size,
                        height: p.size * 0.6
                    )

                    context.opacity = opacity * 0.8
                    context.fill(
                        RoundedRectangle(cornerRadius: 1).path(in: rect),
                        with: .color(p.color)
                    )
                }
            }
        }
        .onAppear { generateParticles() }
    }

    private func generateParticles() {
        let colors: [Color] = [
            Color(.systemOrange), Color(.systemPink),
            Color(.systemTeal), Color(.systemGreen),
            Color(.systemYellow), Color(.systemIndigo)
        ]
        particles = (0..<40).map { _ in
            ConfettiParticle(
                startX: Double.random(in: 0.1...0.9),
                startY: Double.random(in: -0.3...0.1),
                speed: Double.random(in: 30...70),
                wobble: Double.random(in: 1.5...4.0),
                size: Double.random(in: 4...8),
                color: colors.randomElement()!,
                delay: Double.random(in: 0...0.6),
                lifetime: Double.random(in: 1.5...2.5)
            )
        }
    }
}

private struct ConfettiParticle {
    let startX: Double
    let startY: Double
    let speed: Double
    let wobble: Double
    let size: Double
    let color: Color
    let delay: Double
    let lifetime: Double
}

#Preview {
    SessionSummarySheet(session: Session(title: "测试 Session", source: .wifi))
        .environmentObject(PersonalizationManager())
}
