import SwiftUI

// MARK: - Section Label

struct FluxSectionLabel: View {
    let title: String
    let icon: String

    var body: some View {
        Label(title, systemImage: icon)
            .font(Flux.Typography.section)
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
            .tracking(1)
    }
}

// MARK: - Metric Card

struct FluxMetricCard: View {
    let title: String
    let value: String
    let icon: String
    var tint: Color = .primary
    var progress: Double?

    var body: some View {
        VStack(spacing: Flux.Spacing.inner) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(tint)

            Text(value)
                .font(Flux.Typography.metric(22))
                .contentTransition(.numericText())

            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .tracking(1)

            if let progress {
                ProgressView(value: min(max(progress, 0), 1))
                    .tint(tint)
            }
        }
        .padding(Flux.Spacing.item)
        .frame(maxWidth: .infinity)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title) \(value)")
    }
}

// MARK: - Status Badge

struct FluxStatusBadge: View {
    let label: String
    let icon: String
    let tint: Color
    var isActive: Bool = true

    var body: some View {
        Label(label, systemImage: icon)
            .font(.caption)
            .fontWeight(.semibold)
            .foregroundStyle(tint)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(tint.opacity(0.12), in: Capsule())
            .symbolEffect(.pulse, isActive: isActive)
    }
}

// MARK: - EMG Bar Chart

struct FluxEMGBars: View {
    let rms: [Double]
    var height: CGFloat = 60
    var barColor: Color = Flux.Colors.accent

    var body: some View {
        HStack(alignment: .bottom, spacing: 4) {
            let maxVal = max(rms.max() ?? 1, 1)
            ForEach(Array(rms.enumerated()), id: \.offset) { idx, val in
                VStack(spacing: 2) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(barColor.gradient)
                        .frame(height: max(4, CGFloat(val / maxVal) * height))
                        .animation(.easeOut(duration: 0.15), value: val)

                    Text("C\(idx + 1)")
                        .font(.system(size: 8, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
                .frame(maxWidth: .infinity)
            }
        }
        .frame(height: height + 16)
    }
}

// MARK: - Live Indicator

struct FluxLiveIndicator: View {
    let isLive: Bool

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(isLive ? .green : .red)
                .frame(width: 8, height: 8)
            Text(isLive ? "LIVE" : "OFF")
                .font(Flux.Typography.monoSmall)
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Card Container

struct FluxCard<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        content
            .padding()
            .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.large))
    }
}

// MARK: - Share Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_: UIActivityViewController, context: Context) {}
}
