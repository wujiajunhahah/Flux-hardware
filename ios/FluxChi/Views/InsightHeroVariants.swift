import SwiftUI

// 5 个 Hero variant — DailyInsightHeroCard 内 swipe 切换。
// 系统按 InsightStats 数据状态选择默认值（见 InsightStats.defaultVariant）。

// MARK: - Hero/Streak

struct HeroStreakView: View {
    let stats: InsightStats
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 18) {
                // Left: dial + big number
                VStack(alignment: .leading, spacing: 8) {
                    StaminaDial(
                        value: stats.avgStamina,
                        tint: tint
                    )
                    .frame(width: 96, height: 96)

                    Text(stats.headline)
                        .font(.system(size: 17, weight: .semibold, design: .rounded))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                }

                // Right: sub data
                VStack(alignment: .leading, spacing: 14) {
                    metric(icon: "timer", label: "今日累计", value: "\(stats.totalMinutes) 分钟")
                    metric(icon: "rectangle.stack.fill", label: "场次", value: "\(stats.today.count)")
                    if let delta = stats.deltaVsYesterday, abs(delta) >= 3 {
                        TrendBadge(delta: delta)
                    }
                    if let best = stats.bestSlot, best.1 > 50 {
                        chip(icon: best.0.iconName, text: "\(best.0.rawValue)最佳", tint: Flux.Colors.success)
                    }
                    Spacer(minLength: 0)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(cardBackground)
        }
        .buttonStyle(.plain)
    }

    private var tint: Color {
        Flux.Colors.forStaminaValue(stats.avgStamina)
    }

    private func metric(icon: String, label: String, value: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
                .frame(width: 14)
            Text(label)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
            Spacer(minLength: 0)
            Text(value)
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundStyle(.primary)
                .contentTransition(.numericText())
        }
    }

    private func chip(icon: String, text: String, tint: Color) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 10))
            Text(text)
                .font(.system(size: 11, weight: .medium))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(tint.opacity(0.10), in: Capsule())
    }

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
            .fill(Color(.secondarySystemGroupedBackground))
            .overlay(
                RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [tint.opacity(0.06), .clear],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            )
    }
}

// MARK: - Hero/Quote

struct HeroQuoteView: View {
    let stats: InsightStats
    let narrative: String?
    let isLoading: Bool
    let onTap: () -> Void

    private var text: String { narrative ?? stats.fallbackQuote }

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Image(systemName: "quote.opening")
                        .font(.system(size: 26, weight: .semibold))
                        .foregroundStyle(Flux.Colors.accent.opacity(0.7))
                    Spacer()
                    if isLoading {
                        ProgressView().controlSize(.small)
                    }
                }
                Text(text)
                    .font(.system(size: 17, weight: .medium))
                    .foregroundStyle(.primary)
                    .lineSpacing(6)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
                HStack(spacing: 6) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 10))
                    Text(stats.hasData ? "FocuX 今日观察" : "FocuX")
                        .font(.system(size: 11, weight: .medium))
                }
                .foregroundStyle(.tertiary)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(quoteBackground)
        }
        .buttonStyle(.plain)
    }

    private var quoteBackground: some View {
        RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
            .fill(Color(.secondarySystemGroupedBackground))
    }
}

// MARK: - Hero/Compare

struct HeroCompareView: View {
    let stats: InsightStats
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 6) {
                    Image(systemName: "chart.bar.xaxis")
                        .font(.system(size: 11))
                    Text("今 vs 昨")
                        .font(.system(size: 11, weight: .semibold))
                        .tracking(0.5)
                }
                .foregroundStyle(.tertiary)

                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("\(Int(stats.avgStamina))")
                        .font(.system(size: 56, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)
                        .contentTransition(.numericText(value: stats.avgStamina))
                    if let delta = stats.deltaVsYesterday {
                        TrendBadge(delta: delta)
                            .padding(.bottom, 6)
                    }
                }
                .padding(.vertical, -4)

                bars
                Spacer(minLength: 0)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
            )
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var bars: some View {
        let yesterday = stats.previousDayAvg ?? 0
        let today = stats.avgStamina
        let maxV = max(max(yesterday, today), 1)

        VStack(spacing: 10) {
            comparisonRow(label: "昨天", value: yesterday, ratio: yesterday / maxV, tint: .gray)
            comparisonRow(label: "今天", value: today, ratio: today / maxV, tint: Flux.Colors.forStaminaValue(today))
        }
    }

    private func comparisonRow(label: String, value: Double, ratio: Double, tint: Color) -> some View {
        HStack(spacing: 10) {
            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: 32, alignment: .leading)

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 5)
                        .fill(tint.opacity(0.12))
                        .frame(height: 14)
                    RoundedRectangle(cornerRadius: 5)
                        .fill(tint.gradient)
                        .frame(width: geo.size.width * max(0.04, ratio), height: 14)
                }
            }
            .frame(height: 14)

            Text("\(Int(value))")
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundStyle(.primary)
                .frame(width: 32, alignment: .trailing)
        }
    }
}

// MARK: - Hero/Empty

struct HeroEmptyView: View {
    let stats: InsightStats
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 14) {
                Image(systemName: "moon.stars")
                    .font(.system(size: 30, weight: .light))
                    .foregroundStyle(Flux.Colors.accent.opacity(0.6))
                    .symbolEffect(.pulse.byLayer, options: .repeating)

                Text(stats.headline)
                    .font(.system(size: 28, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary)

                Text(stats.emptyInvitation)
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
                    .lineSpacing(4)
                    .fixedSize(horizontal: false, vertical: true)

                Spacer(minLength: 0)

                if let weekly = stats.weeklyAvg, weekly > 0 {
                    HStack(spacing: 8) {
                        Image(systemName: "calendar.badge.checkmark")
                            .font(.system(size: 10))
                        Text("过去 7 天平均续航 \(Int(weekly))")
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                    }
                    .foregroundStyle(.tertiary)
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Hero/Critical

struct HeroCriticalView: View {
    let stats: InsightStats
    let narrative: String?
    let onTap: () -> Void

    private var text: String {
        narrative ?? stats.fallbackQuote
    }

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.circle.fill")
                        .font(.system(size: 14))
                        .foregroundStyle(Flux.Colors.warning)
                        .symbolEffect(.pulse, options: .repeating)
                    Text("需要关注")
                        .font(.system(size: 12, weight: .semibold))
                        .tracking(0.5)
                        .foregroundStyle(Flux.Colors.warning)
                    Spacer()
                    Text("\(Int(stats.avgStamina))")
                        .font(.system(size: 22, weight: .bold, design: .rounded))
                        .foregroundStyle(Flux.Colors.warning)
                        .contentTransition(.numericText(value: stats.avgStamina))
                }

                Text(text)
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(.primary)
                    .lineSpacing(5)
                    .fixedSize(horizontal: false, vertical: true)

                Spacer(minLength: 0)

                HStack(spacing: 8) {
                    if stats.avgFatigue > 0.5 {
                        signalChip(label: "疲劳 \(Int(stats.avgFatigue * 100))%", tint: Flux.Colors.warning)
                    }
                    if stats.avgTension > 0.5 {
                        signalChip(label: "紧张 \(Int(stats.avgTension * 100))%", tint: Color(.systemOrange))
                    }
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
                    .overlay(
                        RoundedRectangle(cornerRadius: Flux.Radius.large, style: .continuous)
                            .stroke(Flux.Colors.warning.opacity(0.3), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(.plain)
    }

    private func signalChip(label: String, tint: Color) -> some View {
        Text(label)
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(tint)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(tint.opacity(0.12), in: Capsule())
    }
}
