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
    var compact: Bool = false

    var body: some View {
        if compact {
            compactLayout
        } else {
            standardLayout
        }
    }

    private var standardLayout: some View {
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

    private var compactLayout: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(size: 14, weight: .bold, design: .rounded))
                .foregroundStyle(tint)
            Text(title)
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
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
    var hideDeadChannels: Bool = true

    private var activeChannels: [(index: Int, value: Double)] {
        rms.enumerated().compactMap { idx, val in
            if hideDeadChannels && val.isZero { return nil }
            return (idx, val)
        }
    }

    var body: some View {
        let channels = activeChannels.isEmpty ? Array(rms.enumerated().map { ($0.offset, $0.element) }) : activeChannels
        let maxVal = max(channels.map(\.value).max() ?? 1, 1)

        HStack(alignment: .bottom, spacing: 4) {
            ForEach(channels, id: \.index) { idx, val in
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
        Circle()
            .fill(isLive ? Flux.Colors.success : Color.primary.opacity(0.2))
            .frame(width: 8, height: 8)
            .shadow(color: isLive ? Flux.Colors.success.opacity(0.5) : .clear, radius: 4)
            .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true),
                        value: isLive)
            .accessibilityLabel(isLive ? "已连接" : "未连接")
    }
}

// MARK: - Card Container

enum FluxCardStyle {
    case standard    // ultraThinMaterial
    case elevated    // secondarySystemGroupedBackground
    case glass       // thinMaterial + 更强模糊
}

struct FluxCard<Content: View>: View {
    var style: FluxCardStyle = .standard
    @ViewBuilder let content: Content

    var body: some View {
        content
            .padding()
            .background {
                RoundedRectangle(cornerRadius: Flux.Radius.large)
                    .fill(backgroundForStyle)
            }
    }

    private var backgroundForStyle: AnyShapeStyle {
        switch style {
        case .standard:
            AnyShapeStyle(.ultraThinMaterial)
        case .elevated:
            AnyShapeStyle(Color(.secondarySystemGroupedBackground))
        case .glass:
            AnyShapeStyle(.thinMaterial)
        }
    }
}

// MARK: - Empty State

struct FluxEmptyState: View {
    let title: String
    let message: String
    let icon: String
    var action: (() -> Void)?
    var actionLabel: String?

    var body: some View {
        ContentUnavailableView {
            Label(title, systemImage: icon)
        } description: {
            Text(message)
        } actions: {
            if let action, let label = actionLabel {
                Button(action: action) {
                    Text(label)
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
}

// MARK: - Share Helper

enum FluxShare {
    static func present(items: [Any]) {
        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let root = windowScene.windows.first?.rootViewController else { return }
        var top = root
        while let presented = top.presentedViewController { top = presented }
        let vc = UIActivityViewController(activityItems: items, applicationActivities: nil)
        vc.popoverPresentationController?.sourceView = top.view
        vc.popoverPresentationController?.sourceRect = CGRect(
            x: top.view.bounds.midX, y: top.view.bounds.midY, width: 0, height: 0
        )
        top.present(vc, animated: true)
    }

    static func shareFile(url: URL) {
        present(items: [url])
    }
}
