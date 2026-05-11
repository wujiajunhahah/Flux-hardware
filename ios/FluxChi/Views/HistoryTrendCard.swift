import SwiftUI
import Charts

/// 续航 28 天趋势卡。Whoop 式渐变填充折线，强调长期变化方向。
struct HistoryTrendCard: View {

    struct DayPoint: Identifiable, Hashable {
        let id = UUID()
        let date: Date
        let avgStamina: Double
        let hasData: Bool
    }

    let points: [DayPoint]

    private var dataOnly: [DayPoint] { points.filter(\.hasData) }

    private var overallAvg: Double {
        let vals = dataOnly.map(\.avgStamina)
        guard !vals.isEmpty else { return 0 }
        return vals.reduce(0, +) / Double(vals.count)
    }

    /// 趋势描述：比较前 14 天均值 vs 后 14 天均值。
    private var trendDescription: String {
        let recent = dataOnly.suffix(14).map(\.avgStamina)
        let older = dataOnly.prefix(max(0, dataOnly.count - 14)).map(\.avgStamina)
        guard !recent.isEmpty, !older.isEmpty else { return "数据积累中" }
        let recentAvg = recent.reduce(0, +) / Double(recent.count)
        let olderAvg = older.reduce(0, +) / Double(older.count)
        let delta = recentAvg - olderAvg
        if delta > 3 { return String(format: "比上半月高 %.0f", delta) }
        if delta < -3 { return String(format: "比上半月低 %.0f", abs(delta)) }
        return "保持稳定"
    }

    private var accentColor: Color {
        Flux.Colors.forStaminaValue(overallAvg)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            chart
            footer
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
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.tertiary)
                Text("续航趋势")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.tertiary)
                    .textCase(.uppercase)
                    .tracking(0.8)
            }

            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("\(Int(overallAvg))")
                    .font(.system(size: 32, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)
                    .contentTransition(.numericText())
                Text("过去 28 天平均")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Chart

    private var chart: some View {
        Chart {
            ForEach(points) { point in
                if point.hasData {
                    AreaMark(
                        x: .value("日期", point.date),
                        y: .value("续航", point.avgStamina)
                    )
                    .foregroundStyle(
                        LinearGradient(
                            colors: [accentColor.opacity(0.35), accentColor.opacity(0.02)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .interpolationMethod(.catmullRom)

                    LineMark(
                        x: .value("日期", point.date),
                        y: .value("续航", point.avgStamina)
                    )
                    .foregroundStyle(accentColor)
                    .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                    .interpolationMethod(.catmullRom)
                }
            }

            RuleMark(y: .value("平均", overallAvg))
                .foregroundStyle(Color.primary.opacity(0.15))
                .lineStyle(StrokeStyle(lineWidth: 0.5, dash: [3, 3]))
        }
        .chartYScale(domain: 0...100)
        .chartYAxis(.hidden)
        .chartXAxis {
            AxisMarks(values: .stride(by: .day, count: 7)) { _ in
                AxisValueLabel(format: .dateTime.day().month(.abbreviated), centered: false)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(Color.secondary)
            }
        }
        .frame(height: 120)
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(accentColor)
                .frame(width: 6, height: 6)
            Text(trendDescription)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
            Spacer()
            Text("\(dataOnly.count) 天有数据")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.tertiary)
        }
    }
}
