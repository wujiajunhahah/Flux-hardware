import SwiftUI
import Charts

/// 本周专注英雄卡。Oura 风格大数字 + 7 天柱状条；柱高 = 总时长，柱色 = 平均续航。
struct HistoryWeekHero: View {

    struct DayBar: Identifiable, Hashable {
        let id = UUID()
        let date: Date
        let weekdayShort: String  // 一二三四五六日
        let totalMinutes: Double
        let avgStamina: Double    // 0-100，0 表示无数据
        let isToday: Bool
    }

    let bars: [DayBar]
    let weekLabel: String
    let totalSeconds: TimeInterval
    let sessionCount: Int
    let avgStamina: Double
    let longestStreakMin: Int  // 单次最长持续工作

    private var maxMinutes: Double {
        max(bars.map(\.totalMinutes).max() ?? 1, 1)
    }

    private var totalLabel: (value: String, unit: String) {
        let h = Int(totalSeconds) / 3600
        let m = (Int(totalSeconds) % 3600) / 60
        if h > 0 {
            return ("\(h):\(String(format: "%02d", m))", "h")
        }
        return ("\(m)", "min")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            barChart
            metricChips
        }
        .padding(20)
        .background {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(.ultraThinMaterial)
        }
        .overlay {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.06), lineWidth: 0.5)
        }
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.tertiary)
                Text(weekLabel)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.tertiary)
                    .textCase(.uppercase)
                    .tracking(0.8)
            }

            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(totalLabel.value)
                    .font(.system(size: 44, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)
                    .contentTransition(.numericText())
                Text(totalLabel.unit)
                    .font(.system(size: 18, weight: .semibold, design: .rounded))
                    .foregroundStyle(.secondary)
                    .padding(.leading, 2)
            }
            .padding(.top, -2)

            Text("本周专注")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Bar Chart

    private var barChart: some View {
        HStack(alignment: .bottom, spacing: 8) {
            ForEach(bars) { bar in
                VStack(spacing: 8) {
                    GeometryReader { geo in
                        VStack {
                            Spacer(minLength: 0)
                            RoundedRectangle(cornerRadius: 4, style: .continuous)
                                .fill(colorFor(bar))
                                .frame(height: max(geo.size.height * CGFloat(bar.totalMinutes / maxMinutes), bar.totalMinutes > 0 ? 4 : 2))
                                .opacity(bar.totalMinutes > 0 ? 1 : 0.25)
                        }
                    }
                    .frame(maxWidth: .infinity)

                    Text(bar.weekdayShort)
                        .font(.system(size: 11, weight: bar.isToday ? .bold : .medium))
                        .foregroundStyle(bar.isToday ? AnyShapeStyle(Color.primary) : AnyShapeStyle(HierarchicalShapeStyle.tertiary))
                }
            }
        }
        .frame(height: 90)
    }

    private func colorFor(_ bar: DayBar) -> Color {
        guard bar.totalMinutes > 0 else { return Color.secondary.opacity(0.3) }
        guard bar.avgStamina > 0 else { return Color.secondary.opacity(0.5) }
        return Flux.Colors.forStaminaValue(bar.avgStamina)
    }

    // MARK: - Metric Chips

    private var metricChips: some View {
        HStack(spacing: 12) {
            metricChip(
                icon: "bolt.fill",
                value: avgStamina > 0 ? "\(Int(avgStamina))" : "—",
                label: "平均续航",
                tint: avgStamina > 0 ? Flux.Colors.forStaminaValue(avgStamina) : .secondary
            )
            metricChip(
                icon: "number",
                value: "\(sessionCount)",
                label: "场次",
                tint: .secondary
            )
            metricChip(
                icon: "timer",
                value: longestStreakMin > 0 ? "\(longestStreakMin)m" : "—",
                label: "最长持续",
                tint: .secondary
            )
        }
    }

    private func metricChip(icon: String, value: String, label: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(tint)
                Text(label)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.tertiary)
                    .textCase(.uppercase)
                    .tracking(0.5)
            }
            Text(value)
                .font(.system(size: 17, weight: .semibold, design: .rounded))
                .foregroundStyle(.primary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.primary.opacity(0.04))
        }
    }
}
