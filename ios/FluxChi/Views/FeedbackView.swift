import SwiftUI
import SwiftData

struct FeedbackView: View {
    let session: Session
    @Environment(\.modelContext) private var modelContext
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var personalization: PersonalizationManager

    @State private var feeling: UserFeeling = .okay
    @State private var accuracy: Int = 3
    @State private var notes: String = ""

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: Flux.Spacing.section) {
                    sessionOverview
                    feelingPicker
                    accuracySection
                    comparisonView
                    notesSection
                }
                .padding()
            }
            .navigationTitle("记录反馈")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("跳过") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("提交") { submitFeedback() }
                        .fontWeight(.semibold)
                }
            }
        }
    }

    // MARK: - Overview

    private var sessionOverview: some View {
        VStack(spacing: 8) {
            Text(session.title)
                .font(.headline)

            HStack(spacing: 16) {
                Label(Flux.formatDuration(session.duration), systemImage: "clock")
                if let avg = session.avgStamina {
                    Label("平均 \(Int(avg))", systemImage: "gauge.with.dots.needle.67percent")
                }
            }
            .font(.subheadline)
            .foregroundStyle(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.large))
    }

    // MARK: - Feeling

    private var feelingPicker: some View {
        VStack(alignment: .leading, spacing: Flux.Spacing.item) {
            Text("这次工作你实际感觉如何？")
                .font(.subheadline)
                .fontWeight(.medium)

            HStack(spacing: 12) {
                ForEach(UserFeeling.allCases) { f in
                    Button {
                        withAnimation(.snappy) { feeling = f }
                    } label: {
                        VStack(spacing: 6) {
                            Image(systemName: f.icon)
                                .font(.title2)
                            Text(f.displayName)
                                .font(.caption)
                        }
                        .foregroundStyle(feeling == f ? .white : f.color)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(
                            feeling == f ? AnyShapeStyle(f.color) : AnyShapeStyle(f.color.opacity(0.1)),
                            in: .rect(cornerRadius: Flux.Radius.medium)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    // MARK: - Accuracy

    private var accuracySection: some View {
        VStack(alignment: .leading, spacing: Flux.Spacing.item) {
            Text("系统的判断准确吗？")
                .font(.subheadline)
                .fontWeight(.medium)

            HStack(spacing: 8) {
                ForEach(1...5, id: \.self) { i in
                    Button {
                        withAnimation(.snappy) { accuracy = i }
                    } label: {
                        Image(systemName: i <= accuracy ? "star.fill" : "star")
                            .font(.title2)
                            .foregroundStyle(i <= accuracy ? .yellow : .secondary)
                    }
                    .buttonStyle(.plain)
                }

                Spacer()

                Text(accuracyLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var accuracyLabel: String {
        switch accuracy {
        case 1:  return "完全不准"
        case 2:  return "不太准"
        case 3:  return "一般"
        case 4:  return "比较准"
        case 5:  return "非常准"
        default: return ""
        }
    }

    // MARK: - Comparison

    @ViewBuilder
    private var comparisonView: some View {
        if let avg = session.avgStamina {
            let predicted = staminaStateFor(avg)

            VStack(alignment: .leading, spacing: Flux.Spacing.item) {
                Text("对比")
                    .font(.subheadline)
                    .fontWeight(.medium)

                HStack {
                    VStack(spacing: 4) {
                        Text("系统预测")
                            .font(.caption)
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
                        .foregroundStyle(.secondary)

                    VStack(spacing: 4) {
                        Text("实际感受")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        FluxStatusBadge(
                            label: feeling.displayName,
                            icon: feeling.icon,
                            tint: feeling.color,
                            isActive: false
                        )
                    }
                    .frame(maxWidth: .infinity)
                }
                .padding()
                .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))
            }
        }
    }

    // MARK: - Notes

    private var notesSection: some View {
        VStack(alignment: .leading, spacing: Flux.Spacing.inner) {
            Text("备注（可选）")
                .font(.subheadline)
                .fontWeight(.medium)

            TextField("例如：今天喝了咖啡所以比较精神…", text: $notes, axis: .vertical)
                .lineLimit(3...6)
                .textFieldStyle(.roundedBorder)
        }
    }

    // MARK: - Actions

    private func submitFeedback() {
        let fb = UserFeedback(feeling: feeling, accuracyRating: accuracy, notes: notes)
        fb.session = session
        session.feedback = fb
        modelContext.insert(fb)
        try? modelContext.save()

        personalization.addTrainingData(session: session, feedback: fb)
        dismiss()
    }

    private func staminaStateFor(_ value: Double) -> StaminaState {
        if value > 60 { return .focused }
        if value > 30 { return .fading }
        return .depleted
    }
}
