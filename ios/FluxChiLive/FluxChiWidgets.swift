import WidgetKit
import SwiftUI
import Charts

// MARK: - Shared Timeline Provider

struct FluxWidgetEntry: TimelineEntry {
    let date: Date
    let snapshot: WidgetDataManager.WidgetSnapshot
}

struct FluxWidgetProvider: TimelineProvider {
    func placeholder(in context: Context) -> FluxWidgetEntry {
        FluxWidgetEntry(date: .now, snapshot: .empty)
    }

    func getSnapshot(in context: Context, completion: @escaping (FluxWidgetEntry) -> Void) {
        let entry = FluxWidgetEntry(date: .now, snapshot: WidgetDataManager.readSnapshot())
        completion(entry)
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<FluxWidgetEntry>) -> Void) {
        let entry = FluxWidgetEntry(date: .now, snapshot: WidgetDataManager.readSnapshot())
        // 每 15 分钟刷新一次（WidgetKit 最小间隔）
        let nextUpdate = Calendar.current.date(byAdding: .minute, value: 15, to: .now) ?? .now
        let timeline = Timeline(entries: [entry], policy: .after(nextUpdate))
        completion(timeline)
    }
}

// MARK: - Widget Definitions

/// 小组件: 今日场次
struct FluxSessionCountWidget: Widget {
    let kind = "FluxSessionCount"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: FluxWidgetProvider()) { entry in
            SessionCountWidgetView(entry: entry)
                .containerBackground(.ultraThinMaterial, for: .widget)
        }
        .configurationDisplayName("今日专注")
        .description("今天的专注场次和累计时间")
        .supportedFamilies([.systemSmall])
    }
}

/// 小组件: 平均续航
struct FluxStaminaWidget: Widget {
    let kind = "FluxStamina"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: FluxWidgetProvider()) { entry in
            StaminaWidgetView(entry: entry)
                .containerBackground(.ultraThinMaterial, for: .widget)
        }
        .configurationDisplayName("平均续航")
        .description("今天的平均续航值")
        .supportedFamilies([.systemSmall])
    }
}

/// 中组件: 续航趋势
struct FluxTrendWidget: Widget {
    let kind = "FluxTrend"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: FluxWidgetProvider()) { entry in
            TrendWidgetView(entry: entry)
                .containerBackground(.ultraThinMaterial, for: .widget)
        }
        .configurationDisplayName("续航趋势")
        .description("近 7 天的续航趋势图")
        .supportedFamilies([.systemMedium])
    }
}

/// 大组件: 综合面板
struct FluxDashboardWidget: Widget {
    let kind = "FluxDashboard"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: FluxWidgetProvider()) { entry in
            DashboardWidgetView(entry: entry)
                .containerBackground(.ultraThinMaterial, for: .widget)
        }
        .configurationDisplayName("FocuX 面板")
        .description("综合续航数据面板")
        .supportedFamilies([.systemLarge])
    }
}

// MARK: - Small Widget View: 今日场次

private struct SessionCountWidgetView: View {
    let entry: FluxWidgetEntry
    private var data: WidgetDataManager.WidgetSnapshot { entry.snapshot }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Image(systemName: "flame.fill")
                    .font(.system(size: 13))
                    .foregroundStyle(.orange)
                Spacer()
                Text("FocuX")
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }

            Spacer()

            Text("\(data.todaySessionCount)")
                .font(.system(size: 44, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            Text("场专注")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)

            Spacer().frame(height: 6)

            Text(data.todayTotalMin > 0 ? "累计 \(data.todayTotalMin) 分钟" : "今日未记录")
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
        }
    }
}

// MARK: - Small Widget View: 平均续航

private struct StaminaWidgetView: View {
    let entry: FluxWidgetEntry
    private var data: WidgetDataManager.WidgetSnapshot { entry.snapshot }
    private var color: Color { staminaColor(data.todayAvgStamina) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Image(systemName: "bolt.fill")
                    .font(.system(size: 13))
                    .foregroundStyle(color)
                Spacer()
                Text("FocuX")
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }

            Spacer()

            if data.todayAvgStamina > 0 {
                Text("\(Int(data.todayAvgStamina))")
                    .font(.system(size: 44, weight: .bold, design: .rounded))
                    .foregroundStyle(color)

                Text("平均续航")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)

                Spacer().frame(height: 6)

                // 迷你进度条
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule()
                            .fill(Color.primary.opacity(0.08))
                            .frame(height: 4)
                        Capsule()
                            .fill(color.gradient)
                            .frame(width: geo.size.width * data.todayAvgStamina / 100, height: 4)
                    }
                }
                .frame(height: 4)
            } else {
                Text("--")
                    .font(.system(size: 44, weight: .bold, design: .rounded))
                    .foregroundStyle(.quaternary)

                Text("平均续航")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.tertiary)

                Spacer().frame(height: 6)

                Text("开始记录后显示")
                    .font(.system(size: 11))
                    .foregroundStyle(.quaternary)
            }
        }
    }
}

// MARK: - Medium Widget View: 7天趋势

private struct TrendWidgetView: View {
    let entry: FluxWidgetEntry
    private var data: WidgetDataManager.WidgetSnapshot { entry.snapshot }
    private var color: Color { staminaColor(data.weeklyAvgStamina) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            headerRow
            if data.weeklyDays.count >= 2 {
                chartView
            } else {
                placeholderView
            }
        }
    }

    private var headerRow: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text("近 7 天")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                HStack(alignment: .firstTextBaseline, spacing: 3) {
                    Text(data.weeklyAvgStamina > 0 ? "\(Int(data.weeklyAvgStamina))" : "--")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundStyle(data.weeklyAvgStamina > 0 ? color : .quaternary)
                    Text("avg")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 1) {
                Text("FocuX")
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.tertiary)
                if data.todaySessionCount > 0 {
                    Text("今日 \(data.todaySessionCount) 次")
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    private var chartView: some View {
        Chart(data.weeklyDays) { day in
            BarMark(
                x: .value("日期", day.date, unit: .day),
                y: .value("续航", day.avgStamina)
            )
            .foregroundStyle(staminaColor(day.avgStamina).gradient)
            .cornerRadius(3)
        }
        .chartYScale(domain: 0...100)
        .chartYAxis(.hidden)
        .chartXAxis {
            AxisMarks(values: .stride(by: .day)) { _ in
                AxisValueLabel(format: .dateTime.weekday(.narrow))
                    .font(.system(size: 9))
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private var placeholderView: some View {
        VStack {
            Spacer()
            Text("记录更多数据后显示趋势图")
                .font(.system(size: 12))
                .foregroundStyle(.quaternary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Large Widget View: 综合面板

private struct DashboardWidgetView: View {
    let entry: FluxWidgetEntry
    private var data: WidgetDataManager.WidgetSnapshot { entry.snapshot }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // 顶部标题
            HStack {
                Text("FocuX")
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                Text("面板")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(.secondary)
                Spacer()
                if let time = data.lastSessionTime {
                    Text(time, format: .dateTime.hour().minute())
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
            }

            // 第一行: 两个 metric
            HStack(spacing: 16) {
                metricBlock(
                    value: "\(data.todaySessionCount)",
                    label: "场专注",
                    sub: "\(data.todayTotalMin)m",
                    color: .orange
                )
                metricBlock(
                    value: data.todayAvgStamina > 0 ? "\(Int(data.todayAvgStamina))" : "--",
                    label: "平均续航",
                    sub: data.bestSlotName ?? "—",
                    color: staminaColor(data.todayAvgStamina)
                )
                metricBlock(
                    value: data.weeklyAvgStamina > 0 ? "\(Int(data.weeklyAvgStamina))" : "--",
                    label: "周均续航",
                    sub: "\(data.weeklyDays.count)天",
                    color: staminaColor(data.weeklyAvgStamina)
                )
            }

            // 分隔线
            Rectangle()
                .fill(.quaternary)
                .frame(height: 0.5)

            // 趋势图
            if data.weeklyDays.count >= 2 {
                VStack(alignment: .leading, spacing: 4) {
                    Text("7 天趋势")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)

                    Chart(data.weeklyDays) { day in
                        LineMark(
                            x: .value("日期", day.date),
                            y: .value("续航", day.avgStamina)
                        )
                        .interpolationMethod(.catmullRom)
                        .foregroundStyle(Color.red.gradient)
                        .lineStyle(StrokeStyle(lineWidth: 2))

                        AreaMark(
                            x: .value("日期", day.date),
                            yStart: .value("底", 0),
                            yEnd: .value("续航", day.avgStamina)
                        )
                        .interpolationMethod(.catmullRom)
                        .foregroundStyle(
                            .linearGradient(
                                colors: [Color.red.opacity(0.2), .clear],
                                startPoint: .top, endPoint: .bottom
                            )
                        )
                    }
                    .chartYScale(domain: 0...100)
                    .chartYAxis(.hidden)
                    .chartXAxis {
                        AxisMarks(values: .stride(by: .day)) { _ in
                            AxisValueLabel(format: .dateTime.weekday(.narrow))
                                .font(.system(size: 9))
                                .foregroundStyle(.tertiary)
                        }
                    }
                }
            } else {
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        VStack(spacing: 6) {
                            Image(systemName: "chart.xyaxis.line")
                                .font(.system(size: 24))
                                .foregroundStyle(.quaternary)
                            Text("完成更多记录后显示趋势")
                                .font(.system(size: 12))
                                .foregroundStyle(.quaternary)
                        }
                        Spacer()
                    }
                    Spacer()
                }
            }

            Spacer(minLength: 0)

            // 底部最佳时段
            if let slot = data.bestSlotName, data.bestSlotAvg > 0 {
                HStack(spacing: 6) {
                    Image(systemName: slotIcon(slot))
                        .font(.system(size: 11))
                        .foregroundStyle(.cyan)
                    Text("最佳时段 \(slot)")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("续航 \(Int(data.bestSlotAvg))")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    private func metricBlock(value: String, label: String, sub: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(.system(size: 26, weight: .bold, design: .rounded))
                .foregroundStyle(color)
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
            Text(sub)
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Helpers

private func staminaColor(_ value: Double) -> Color {
    if value > 60 { return .red }
    if value > 30 { return .orange }
    return .purple
}

private func slotIcon(_ slot: String) -> String {
    switch slot {
    case "上午": return "sunrise.fill"
    case "午间": return "sun.max.fill"
    case "下午": return "sun.haze.fill"
    case "晚间": return "moon.stars.fill"
    default:     return "clock.fill"
    }
}
