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
        let nextUpdate = Calendar.current.date(byAdding: .minute, value: 15, to: .now) ?? .now
        let timeline = Timeline(entries: [entry], policy: .after(nextUpdate))
        completion(timeline)
    }
}

// MARK: - Brand Tokens (与 FluxTokens 对齐)

/// Widget 端轻量色彩系统，与主 App Flux.Colors 暖珊瑚色系对齐
private enum WT {
    // 主色 — 暖珊瑚，与 Flux.Colors.accent 一致
    static let accent = Color(red: 0.85, green: 0.50, blue: 0.38) // #D98061

    // 续航状态色 — 与 Flux.Colors.forStaminaValue 一致
    static func staminaColor(_ value: Double) -> Color {
        if value > 60 { return accent }                                         // 珊瑚 — 专注
        if value > 30 { return Color(red: 0.85, green: 0.68, blue: 0.42) }     // 琥珀 — 衰退
        if value > 0  { return Color(red: 0.62, green: 0.50, blue: 0.64) }     // 薰衣草 — 耗尽
        return .secondary
    }

    static func staminaLabel(_ value: Double) -> String {
        if value > 60 { return "专注" }
        if value > 30 { return "衰退" }
        if value > 0  { return "偏低" }
        return "—"
    }

    // 文字层级
    static let textDim = Color(white: 0.45)
    static let textMuted = Color(white: 0.55)

    // 分隔线
    static let divider = Color.white.opacity(0.06)

    // 时段图标
    static func slotIcon(_ slot: String) -> String {
        switch slot {
        case "上午": return "sunrise.fill"
        case "午间": return "sun.max.fill"
        case "下午": return "sun.haze.fill"
        case "晚间": return "moon.stars.fill"
        default:     return "clock.fill"
        }
    }
}

// MARK: - Widget Definitions

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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Small Widget: 今日场次
// 设计参考 TideGuide：大数字主英雄 + 底部辅助数据行
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

private struct SessionCountWidgetView: View {
    let entry: FluxWidgetEntry
    private var data: WidgetDataManager.WidgetSnapshot { entry.snapshot }

    var body: some View {
        VStack(spacing: 0) {
            // 顶栏：品牌图标 + 标签
            HStack(alignment: .center) {
                Image(systemName: "flame.fill")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(WT.accent)
                Spacer()
                Text("FOCUX")
                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                    .foregroundStyle(WT.textDim)
                    .tracking(1.5)
            }

            Spacer()

            // 主数字 — 信息第一层级
            Text("\(data.todaySessionCount)")
                .font(.system(size: 48, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
                .contentTransition(.numericText())

            // 标签 — 信息第二层级
            Text("场专注")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
                .padding(.top, -4)

            Spacer().frame(height: 10)

            // 底部辅助行 — TideGuide 式图标+数据对
            HStack(spacing: 4) {
                Image(systemName: "clock")
                    .font(.system(size: 9))
                    .foregroundStyle(WT.textDim)
                Text(data.todayTotalMin > 0 ? "\(data.todayTotalMin)min" : "未记录")
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(WT.textDim)
            }
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Small Widget: 平均续航（弧形指示器）
// 设计参考 TideGuide：圆弧进度 + 中心大数字 + 底部状态
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

private struct StaminaWidgetView: View {
    let entry: FluxWidgetEntry
    private var data: WidgetDataManager.WidgetSnapshot { entry.snapshot }
    private var color: Color { WT.staminaColor(data.todayAvgStamina) }
    private var progress: Double { min(data.todayAvgStamina / 100.0, 1.0) }
    private var hasData: Bool { data.todayAvgStamina > 0 }

    var body: some View {
        VStack(spacing: 0) {
            // 顶栏
            HStack {
                Text("续航")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("FOCUX")
                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                    .foregroundStyle(WT.textDim)
                    .tracking(1.5)
            }

            Spacer()

            // 弧形进度环 — 3/4 圆弧，仿 TideGuide 曲线数据呈现
            ZStack {
                // 背景弧
                Circle()
                    .trim(from: 0, to: 0.75)
                    .stroke(
                        Color.primary.opacity(0.06),
                        style: StrokeStyle(lineWidth: 5, lineCap: .round)
                    )
                    .rotationEffect(.degrees(135))

                // 进度弧
                if hasData {
                    Circle()
                        .trim(from: 0, to: 0.75 * progress)
                        .stroke(
                            AngularGradient(
                                colors: [color.opacity(0.3), color],
                                center: .center,
                                startAngle: .degrees(135),
                                endAngle: .degrees(135 + 270 * progress)
                            ),
                            style: StrokeStyle(lineWidth: 5, lineCap: .round)
                        )
                        .rotationEffect(.degrees(135))
                }

                // 中心数值
                VStack(spacing: 0) {
                    Text(hasData ? "\(Int(data.todayAvgStamina))" : "--")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundStyle(hasData ? AnyShapeStyle(color) : AnyShapeStyle(.quaternary))
                        .contentTransition(.numericText())
                    Text("avg")
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(WT.textDim)
                }
            }
            .frame(width: 84, height: 84)

            Spacer()

            // 底部状态
            if hasData {
                HStack(spacing: 4) {
                    Circle()
                        .fill(color)
                        .frame(width: 5, height: 5)
                    Text(WT.staminaLabel(data.todayAvgStamina))
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            } else {
                Text("等待数据")
                    .font(.system(size: 10))
                    .foregroundStyle(.quaternary)
            }
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Medium Widget: 7天趋势（流线曲线）
// 设计参考 TideGuide：平滑曲线主视觉 + 数据点标记 + 渐变面积
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

private struct TrendWidgetView: View {
    let entry: FluxWidgetEntry
    private var data: WidgetDataManager.WidgetSnapshot { entry.snapshot }
    private var color: Color { WT.staminaColor(data.weeklyAvgStamina) }
    private var hasChart: Bool { data.weeklyDays.count >= 2 }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            headerRow
            if hasChart {
                curveChart
            } else {
                emptyState
            }
        }
    }

    // ── 信息第一层：统计摘要 ──
    private var headerRow: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 1) {
                Text("近 7 天")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(WT.textDim)
                    .tracking(1)

                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(data.weeklyAvgStamina > 0 ? "\(Int(data.weeklyAvgStamina))" : "--")
                        .font(.system(size: 32, weight: .bold, design: .rounded))
                        .foregroundStyle(data.weeklyAvgStamina > 0 ? AnyShapeStyle(color) : AnyShapeStyle(.quaternary))
                        .contentTransition(.numericText())

                    Text("avg")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(WT.textDim)
                }
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text("FOCUX")
                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                    .foregroundStyle(WT.textDim)
                    .tracking(1.5)
                if data.todaySessionCount > 0 {
                    HStack(spacing: 3) {
                        Circle()
                            .fill(WT.accent)
                            .frame(width: 4, height: 4)
                        Text("今日 \(data.todaySessionCount) 次")
                            .font(.system(size: 9, weight: .medium))
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    // ── 信息第二层：流线曲线主视觉 ──
    private var curveChart: some View {
        Chart(data.weeklyDays) { day in
            // 渐变面积 — TideGuide 式波浪填充
            AreaMark(
                x: .value("日期", day.date, unit: .day),
                yStart: .value("底", 0),
                yEnd: .value("续航", day.avgStamina)
            )
            .interpolationMethod(.catmullRom)
            .foregroundStyle(
                .linearGradient(
                    colors: [
                        WT.staminaColor(data.weeklyAvgStamina).opacity(0.2),
                        WT.staminaColor(data.weeklyAvgStamina).opacity(0.01)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )

            // 主曲线 — 平滑线
            LineMark(
                x: .value("日期", day.date, unit: .day),
                y: .value("续航", day.avgStamina)
            )
            .interpolationMethod(.catmullRom)
            .foregroundStyle(WT.staminaColor(day.avgStamina))
            .lineStyle(StrokeStyle(lineWidth: 2.5, lineCap: .round))

            // 外圈数据点 — TideGuide 式标记
            PointMark(
                x: .value("日期", day.date, unit: .day),
                y: .value("续航", day.avgStamina)
            )
            .symbolSize(16)
            .foregroundStyle(WT.staminaColor(day.avgStamina))

            // 内圈（挖空效果）
            PointMark(
                x: .value("日期", day.date, unit: .day),
                y: .value("续航", day.avgStamina)
            )
            .symbolSize(5)
            .foregroundStyle(.background)
        }
        .chartYScale(domain: 0...100)
        .chartYAxis(.hidden)
        .chartXAxis {
            AxisMarks(values: .stride(by: .day, count: 1)) { _ in
                AxisValueLabel(format: .dateTime.weekday(.narrow))
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(WT.textDim)
            }
        }
    }

    private var emptyState: some View {
        VStack {
            Spacer()
            HStack {
                Spacer()
                VStack(spacing: 6) {
                    Image(systemName: "waveform.path")
                        .font(.system(size: 20, weight: .light))
                        .foregroundStyle(.quaternary)
                    Text("记录更多数据后显示趋势")
                        .font(.system(size: 11))
                        .foregroundStyle(.quaternary)
                }
                Spacer()
            }
            Spacer()
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Large Widget: 综合面板
// 设计参考 TideGuide：分层信息架构
//  第一层 → 标题 + 日期上下文
//  第二层 → 三列指标卡（图标+数字+标签+细节）
//  第三层 → 英雄曲线（渐变面积 + 平滑线 + 数据点）
//  第四层 → 底栏上下文信息（最佳时段）
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

private struct DashboardWidgetView: View {
    let entry: FluxWidgetEntry
    private var data: WidgetDataManager.WidgetSnapshot { entry.snapshot }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // ── 第一层：标题栏 ──
            titleRow
                .padding(.bottom, 14)

            // ── 第二层：三列指标 ──
            metricsRow
                .padding(.bottom, 12)

            // ── 分隔 ──
            Rectangle()
                .fill(WT.divider)
                .frame(height: 0.5)
                .padding(.bottom, 10)

            // ── 第三层：英雄曲线 ──
            if data.weeklyDays.count >= 2 {
                trendSection
            } else {
                emptyChart
            }

            Spacer(minLength: 0)

            // ── 第四层：底栏 ──
            bottomBar
        }
    }

    // MARK: Title

    private var titleRow: some View {
        HStack(alignment: .center) {
            HStack(spacing: 6) {
                // 品牌标识 — 红色小圆点
                Circle()
                    .fill(WT.accent)
                    .frame(width: 7, height: 7)
                Text("FocuX")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)
            }
            Spacer()
            // 日期上下文 — TideGuide 式右上角时间
            VStack(alignment: .trailing, spacing: 0) {
                Text(entry.date, format: .dateTime.month(.abbreviated).day())
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                if let time = data.lastSessionTime {
                    Text("更新 \(time, format: .dateTime.hour().minute())")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(WT.textDim)
                }
            }
        }
    }

    // MARK: Metrics

    private var metricsRow: some View {
        HStack(spacing: 0) {
            metricCell(
                icon: "flame.fill",
                iconColor: .orange,
                value: "\(data.todaySessionCount)",
                label: "场专注",
                detail: "\(data.todayTotalMin)m"
            )

            Rectangle()
                .fill(WT.divider)
                .frame(width: 0.5, height: 36)

            metricCell(
                icon: "bolt.fill",
                iconColor: WT.staminaColor(data.todayAvgStamina),
                value: data.todayAvgStamina > 0 ? "\(Int(data.todayAvgStamina))" : "--",
                label: "日均续航",
                detail: WT.staminaLabel(data.todayAvgStamina)
            )

            Rectangle()
                .fill(WT.divider)
                .frame(width: 0.5, height: 36)

            metricCell(
                icon: "chart.line.uptrend.xyaxis",
                iconColor: WT.staminaColor(data.weeklyAvgStamina),
                value: data.weeklyAvgStamina > 0 ? "\(Int(data.weeklyAvgStamina))" : "--",
                label: "周均续航",
                detail: "\(data.weeklyDays.count)天"
            )
        }
    }

    private func metricCell(icon: String, iconColor: Color, value: String, label: String, detail: String) -> some View {
        VStack(spacing: 3) {
            Image(systemName: icon)
                .font(.system(size: 10))
                .foregroundStyle(iconColor)
            Text(value)
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
                .contentTransition(.numericText())
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
            Text(detail)
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(WT.textDim)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: Trend

    private var trendSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("7 天趋势")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(WT.textDim)
                .tracking(0.8)

            Chart(data.weeklyDays) { day in
                // 渐变面积
                AreaMark(
                    x: .value("日期", day.date, unit: .day),
                    yStart: .value("底", 0),
                    yEnd: .value("续航", day.avgStamina)
                )
                .interpolationMethod(.catmullRom)
                .foregroundStyle(
                    .linearGradient(
                        colors: [WT.accent.opacity(0.18), WT.accent.opacity(0.01)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )

                // 主曲线
                LineMark(
                    x: .value("日期", day.date, unit: .day),
                    y: .value("续航", day.avgStamina)
                )
                .interpolationMethod(.catmullRom)
                .foregroundStyle(WT.accent.gradient)
                .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round))

                // 数据点
                PointMark(
                    x: .value("日期", day.date, unit: .day),
                    y: .value("续航", day.avgStamina)
                )
                .symbolSize(14)
                .foregroundStyle(WT.accent)

                // 内圈
                PointMark(
                    x: .value("日期", day.date, unit: .day),
                    y: .value("续航", day.avgStamina)
                )
                .symbolSize(5)
                .foregroundStyle(.background)
            }
            .chartYScale(domain: 0...100)
            .chartYAxis(.hidden)
            .chartXAxis {
                AxisMarks(values: .stride(by: .day, count: 1)) { _ in
                    AxisValueLabel(format: .dateTime.weekday(.narrow))
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(WT.textDim)
                }
            }
        }
    }

    private var emptyChart: some View {
        VStack {
            Spacer()
            HStack {
                Spacer()
                VStack(spacing: 8) {
                    Image(systemName: "waveform.path")
                        .font(.system(size: 24, weight: .ultraLight))
                        .foregroundStyle(.quaternary)
                    Text("完成更多记录后显示趋势")
                        .font(.system(size: 11))
                        .foregroundStyle(.quaternary)
                }
                Spacer()
            }
            Spacer()
        }
    }

    // MARK: Bottom Bar

    private var bottomBar: some View {
        Group {
            if let slot = data.bestSlotName, data.bestSlotAvg > 0 {
                HStack(spacing: 0) {
                    HStack(spacing: 5) {
                        Image(systemName: WT.slotIcon(slot))
                            .font(.system(size: 10))
                            .foregroundStyle(WT.accent)
                        Text(slot)
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.secondary)
                        Text("最佳时段")
                            .font(.system(size: 10))
                            .foregroundStyle(WT.textDim)
                    }
                    Spacer()
                    Text("\(Int(data.bestSlotAvg))")
                        .font(.system(size: 12, weight: .bold, design: .monospaced))
                        .foregroundStyle(WT.staminaColor(data.bestSlotAvg))
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(.ultraThinMaterial)
                )
            }
        }
    }
}
