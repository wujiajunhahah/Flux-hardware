import SwiftUI

// MARK: - Share Card Sheet

/// 分享今日洞察卡片：渲染 `ShareCardContent` 为 PNG（`ImageRenderer`），通过 `ShareLink` 分享。
struct ShareCardSheet: View {
    let stats: InsightStats
    let narrative: String?

    @Environment(\.dismiss) private var dismiss
    @State private var renderedImage: Image?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    cardPreview
                        .padding(.horizontal, 24)
                        .padding(.top, 16)

                    if let img = renderedImage {
                        ShareLink(
                            item: img,
                            preview: SharePreview("FocuX 今日洞察", image: img)
                        ) {
                            HStack(spacing: 8) {
                                Image(systemName: "square.and.arrow.up")
                                Text("分享")
                                    .font(.system(size: 16, weight: .semibold))
                            }
                            .foregroundStyle(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Flux.Colors.accent, in: Capsule())
                        }
                    } else {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small)
                            Text("生成图片中…")
                                .font(.system(size: 13))
                                .foregroundStyle(.tertiary)
                        }
                    }
                }
                .padding(.bottom, 32)
            }
            .background(Color(.systemGroupedBackground).ignoresSafeArea())
            .navigationTitle("分享今日卡片")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
        .onAppear {
            render()
        }
    }

    @ViewBuilder
    private var cardPreview: some View {
        ShareCardContent(stats: stats, narrative: narrative)
            .shadow(color: .black.opacity(0.15), radius: 16, y: 6)
    }

    @MainActor
    private func render() {
        let renderer = ImageRenderer(content: ShareCardContent(stats: stats, narrative: narrative).frame(width: 380))
        renderer.scale = UIScreen.main.scale
        if let ui = renderer.uiImage {
            renderedImage = Image(uiImage: ui)
        }
    }
}

// MARK: - Share Card Content

/// 分享卡片视觉本体（暗色渐变 + 续航环 + 指标 + 站点）。
struct ShareCardContent: View {
    let stats: InsightStats
    let narrative: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                HStack(spacing: 6) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 11, weight: .semibold))
                    Text("FocuX 今日")
                        .font(.system(size: 11, weight: .semibold))
                        .tracking(0.8)
                }
                .foregroundStyle(.white.opacity(0.85))
                Spacer()
                Text(Date().formatted(.dateTime.month().day()))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.7))
            }

            HStack(alignment: .top, spacing: 16) {
                ZStack {
                    Circle()
                        .stroke(Color.white.opacity(0.2), lineWidth: 7)
                    Circle()
                        .trim(from: 0, to: max(0.04, min(1, stats.avgStamina / 100)))
                        .stroke(Color.white, style: StrokeStyle(lineWidth: 7, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                    Text("\(Int(stats.avgStamina))")
                        .font(.system(size: 30, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                }
                .frame(width: 100, height: 100)

                VStack(alignment: .leading, spacing: 8) {
                    Text(stats.headline)
                        .font(.system(size: 22, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white)
                    Text(stats.subLine)
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.75))
                    if let delta = stats.deltaVsYesterday, abs(delta) >= 3 {
                        Text(delta > 0 ? "比昨天 +\(delta)" : "比昨天 \(delta)")
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.85))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color.white.opacity(0.15), in: Capsule())
                    }
                    Spacer(minLength: 0)
                }
            }

            if let text = narrative ?? Optional(stats.fallbackQuote), !text.isEmpty {
                Text(text)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.white.opacity(0.9))
                    .lineSpacing(5)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 14) {
                shareMetric(label: "紧张", value: "\(Int(stats.avgTension * 100))%")
                shareMetric(label: "疲劳", value: "\(Int(stats.avgFatigue * 100))%")
                shareMetric(label: "一致性", value: "\(Int(stats.avgConsistency * 100))%")
            }

            HStack {
                Text("focux.me")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.white.opacity(0.5))
                Spacer()
                Text("EMG · Vision · Focus")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
        .padding(22)
        .frame(maxWidth: .infinity)
        .background(
            LinearGradient(
                colors: [
                    Color.black,
                    Flux.Colors.accent.opacity(0.55)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 28, style: .continuous)
        )
    }

    private func shareMetric(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .tracking(0.5)
                .foregroundStyle(.white.opacity(0.6))
            Text(value)
                .font(.system(size: 16, weight: .semibold, design: .monospaced))
                .foregroundStyle(.white)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
