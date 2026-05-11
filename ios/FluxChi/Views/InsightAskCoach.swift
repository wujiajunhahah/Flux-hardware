import SwiftUI

// MARK: - AskCoachSheet

/// 提问式洞察 sheet：预设问题 + 自由文本输入，调用 `NLPSummaryEngine.askFollowUp`（iOS 26+）。
/// Detail 视图底部三级链接进入，不作为主入口。
struct AskCoachSheet: View {
    let todaySessions: [Session]
    let recentSessions: [Session]
    let insightText: String?

    @Environment(\.dismiss) private var dismiss
    @State private var messages: [(question: String, answer: String)] = []
    @State private var isLoading = false
    @State private var customQuestion = ""

    private let presets: [String] = [
        "为什么我下午续航总是下降？",
        "怎样延长高效时间？",
        "我的紧张度正常吗？"
    ]

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        if messages.isEmpty {
                            presetSection
                        }
                        messageList
                        if isLoading {
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                Text("思考中…")
                                    .font(.system(size: 13))
                                    .foregroundStyle(.tertiary)
                            }
                            .padding(.horizontal, 20)
                        }
                    }
                    .padding(.vertical, 16)
                }
                .onChange(of: messages.count) { _, _ in
                    withAnimation { proxy.scrollTo(messages.count - 1, anchor: .bottom) }
                }
            }
            .background(Color(.systemGroupedBackground).ignoresSafeArea())
            .safeAreaInset(edge: .bottom) { inputBar }
            .navigationTitle("问问 FocuX")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }

    private var presetSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("常见问题")
                .font(.system(size: 11, weight: .semibold))
                .tracking(1.2)
                .foregroundStyle(.tertiary)
                .padding(.horizontal, 20)
            VStack(spacing: 8) {
                ForEach(presets, id: \.self) { q in
                    Button { ask(question: q) } label: {
                        HStack {
                            Text(q)
                                .font(.system(size: 14))
                                .foregroundStyle(.primary)
                                .multilineTextAlignment(.leading)
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(.tertiary)
                        }
                        .padding(14)
                        .frame(maxWidth: .infinity)
                        .background(
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .fill(Color(.secondarySystemGroupedBackground))
                        )
                    }
                    .buttonStyle(.plain)
                    .disabled(isLoading)
                }
            }
            .padding(.horizontal, 20)
        }
    }

    private var messageList: some View {
        VStack(alignment: .leading, spacing: 18) {
            ForEach(Array(messages.enumerated()), id: \.offset) { idx, msg in
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Spacer(minLength: 40)
                        Text(msg.question)
                            .font(.system(size: 14))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 9)
                            .background(Flux.Colors.accent, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    }
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "sparkles")
                            .font(.system(size: 11))
                            .foregroundStyle(Flux.Colors.accent)
                            .padding(.top, 6)
                        Text(msg.answer)
                            .font(.system(size: 14))
                            .foregroundStyle(.primary)
                            .lineSpacing(5)
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer(minLength: 0)
                    }
                    .padding(.trailing, 40)
                }
                .padding(.horizontal, 20)
                .id(idx)
                .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
        }
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("输入你的问题…", text: $customQuestion)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 14))
                .submitLabel(.send)
                .onSubmit { sendCustom() }
                .disabled(isLoading)
            Button { sendCustom() } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 26))
                    .foregroundStyle(customQuestion.isEmpty ? Color(.tertiaryLabel) : Flux.Colors.accent)
            }
            .disabled(customQuestion.isEmpty || isLoading)
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }

    @MainActor
    private func sendCustom() {
        let q = customQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return }
        customQuestion = ""
        ask(question: q)
    }

    @MainActor
    private func ask(question: String) {
        guard !isLoading else { return }
        isLoading = true
        Task { @MainActor in
            let answer: String
            if #available(iOS 26.0, *) {
                let anomalies = NLPSummaryEngine.shared.detectDailyAnomalies(sessions: todaySessions)
                let weekly = recentSessions.count >= 2
                    ? NLPSummaryEngine.shared.generateWeeklyTrend(sessions: recentSessions)
                    : nil
                let ctx = NLPSummaryEngine.CoachContext(
                    todaySessions: todaySessions,
                    dailyInsight: insightText,
                    anomalies: anomalies,
                    weeklyTrend: weekly
                )
                answer = await NLPSummaryEngine.shared.askFollowUp(context: ctx, question: question)
            } else {
                answer = "需要 iOS 26+ 的 Apple Intelligence 才能给个性化建议。一个通用建议：每 25 分钟主动休息 5 分钟。"
            }
            withAnimation(.spring(response: 0.45, dampingFraction: 0.85)) {
                messages.append((question, answer))
            }
            isLoading = false
        }
    }
}
