import SwiftUI

struct DashboardView: View {
    @EnvironmentObject var service: FluxService

    private var stamina: StaminaData? { service.state?.stamina }
    private var decision: DecisionData? { service.state?.decision }
    private var staminaValue: Double { stamina?.value ?? 0 }
    private var staminaState: StaminaState {
        StaminaState(rawValue: stamina?.state ?? "focused") ?? .focused
    }
    private var rec: Recommendation {
        Recommendation(rawValue: decision?.recommendation ?? "") ?? .keepWorking
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    connectionBanner
                    staminaSection
                    recommendationCard
                    statsRow
                    dimensionsSection
                    activitySection
                    emgSection
                }
                .padding()
            }
            .navigationTitle("FluxChi")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    connectionIndicator
                }
            }
            .refreshable {
                await service.fetchState()
            }
        }
    }

    // MARK: - Connection Banner

    @ViewBuilder
    private var connectionBanner: some View {
        if !service.isConnected {
            HStack(spacing: 12) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text("未连接到服务器")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    if let err = service.connectionError {
                        Text(err)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
                Spacer()
            }
            .padding()
            .background(.orange.opacity(0.1), in: .rect(cornerRadius: 12))
        }
    }

    // MARK: - Stamina Ring

    private var staminaSection: some View {
        StaminaRingView(value: staminaValue, state: staminaState)
            .padding(.vertical, 8)
    }

    // MARK: - Recommendation Card

    @ViewBuilder
    private var recommendationCard: some View {
        if let decision {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Label(rec.displayName, systemImage: rec.systemImage)
                        .font(.headline)
                        .foregroundStyle(urgencyColor(decision.urgency))

                    Spacer()

                    if decision.urgency >= 0.5 {
                        Text("!")
                            .font(.caption)
                            .fontWeight(.bold)
                            .padding(6)
                            .background(urgencyColor(decision.urgency).opacity(0.15))
                            .clipShape(Circle())
                    }
                }

                ForEach(decision.reasons, id: \.self) { reason in
                    Text(reason)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            .padding()
            .background(.ultraThinMaterial, in: .rect(cornerRadius: 16))
        }
    }

    // MARK: - Stats Row

    private var statsRow: some View {
        HStack(spacing: 0) {
            StatCell(
                value: formatMinutes(decision?.continuousWorkMin ?? 0),
                label: "本次",
                icon: "timer"
            )
            Divider().frame(height: 40)
            StatCell(
                value: formatMinutes(decision?.totalWorkMin ?? 0),
                label: "累计",
                icon: "clock.fill"
            )
            Divider().frame(height: 40)
            StatCell(
                value: formatMinutes(
                    staminaState == .recovering
                        ? (stamina?.suggestedBreakMin ?? 0)
                        : (stamina?.suggestedWorkMin ?? 0)
                ),
                label: staminaState == .recovering ? "休息" : "专注",
                icon: staminaState == .recovering ? "moon.fill" : "bolt.fill"
            )
        }
        .padding(.vertical, 12)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: 16))
    }

    // MARK: - Dimensions

    private var dimensionsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("三维度", systemImage: "chart.bar.xaxis")
                .font(.caption)
                .fontWeight(.medium)
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
                .tracking(1)

            HStack(spacing: 12) {
                DimensionCard(
                    title: "专注",
                    value: stamina?.consistency ?? 0,
                    icon: "eye.fill",
                    tint: .blue
                )
                DimensionCard(
                    title: "紧张",
                    value: stamina?.tension ?? 0,
                    icon: "waveform.path.ecg",
                    tint: .orange
                )
                DimensionCard(
                    title: "疲劳",
                    value: stamina?.fatigue ?? 0,
                    icon: "battery.25percent",
                    tint: .red
                )
            }
        }
    }

    // MARK: - Activity

    @ViewBuilder
    private var activitySection: some View {
        if let probs = service.state?.probabilities, !probs.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                Label("活动分类", systemImage: "hand.raised.fingers.spread.fill")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                    .tracking(1)

                let sorted = probs.sorted { $0.value > $1.value }
                ForEach(sorted, id: \.key) { key, value in
                    ActivityRow(name: key, probability: value, isTop: key == sorted.first?.key)
                }
            }
        }
    }

    // MARK: - EMG

    @ViewBuilder
    private var emgSection: some View {
        if let rms = service.state?.rms, !rms.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                Label("EMG 信号", systemImage: "waveform")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                    .tracking(1)

                HStack(alignment: .bottom, spacing: 6) {
                    let maxVal = max(rms.max() ?? 1, 1)
                    ForEach(Array(rms.enumerated()), id: \.offset) { idx, val in
                        VStack(spacing: 4) {
                            RoundedRectangle(cornerRadius: 3)
                                .fill(.red.opacity(0.7))
                                .frame(width: nil, height: max(4, CGFloat(val / maxVal) * 60))
                                .animation(.easeOut(duration: 0.15), value: val)

                            Text("\(idx + 1)")
                                .font(.system(size: 9, weight: .medium, design: .monospaced))
                                .foregroundStyle(.tertiary)
                        }
                        .frame(maxWidth: .infinity)
                    }
                }
                .frame(height: 80)
                .padding()
                .background(.ultraThinMaterial, in: .rect(cornerRadius: 12))
            }
        }
    }

    // MARK: - Helpers

    private var connectionIndicator: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(service.isConnected ? .green : .red)
                .frame(width: 8, height: 8)
            Text(service.isConnected ? "LIVE" : "OFF")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(.secondary)
        }
    }

    private func urgencyColor(_ urgency: Double) -> Color {
        urgency >= 0.7 ? .red : urgency >= 0.5 ? .orange : .primary
    }

    private func formatMinutes(_ m: Double) -> String {
        if m < 1 { return "< 1" }
        return "\(Int(m))"
    }
}

// MARK: - Subviews

private struct StatCell: View {
    let value: String
    let label: String
    let icon: String

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .fontDesign(.rounded)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .tracking(1)
        }
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .combine)
    }
}

private struct DimensionCard: View {
    let title: String
    let value: Double
    let icon: String
    let tint: Color

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(tint)

            Text("\(Int(value * 100))%")
                .font(.title3)
                .fontWeight(.bold)
                .fontDesign(.rounded)
                .contentTransition(.numericText(value: value))

            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .tracking(1)

            ProgressView(value: value)
                .tint(tint)
        }
        .padding(12)
        .frame(maxWidth: .infinity)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: 14))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title) \(Int(value * 100))%")
    }
}

private struct ActivityRow: View {
    let name: String
    let probability: Double
    let isTop: Bool

    var body: some View {
        HStack(spacing: 10) {
            Text(name)
                .font(.system(size: 13, weight: isTop ? .semibold : .regular, design: .monospaced))
                .foregroundStyle(isTop ? .primary : .secondary)
                .frame(width: 80, alignment: .trailing)

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color(.systemGray5))
                    Capsule()
                        .fill(isTop ? .red : Color(.systemGray3))
                        .frame(width: max(2, geo.size.width * probability))
                        .animation(.easeOut(duration: 0.3), value: probability)
                }
            }
            .frame(height: 6)

            Text("\(Int(probability * 100))%")
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 36, alignment: .trailing)
        }
    }
}

#Preview {
    DashboardView()
        .environmentObject(FluxService())
}
