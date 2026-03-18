import SwiftUI
import SwiftData
import Charts

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - FluxCalendarView
// 交互逻辑参考 BIOSORA：日历即数据可视化
//  第一层 → 月份导航 + 英雄统计数字
//  第二层 → 7列日历网格，每格 = stamina 数据圆点
//  第三层 → 选中日的会话卡片列表
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct FluxCalendarView: View {
    @Query(sort: \Session.startedAt, order: .reverse) private var allSessions: [Session]

    @State private var selectedDate: Date = Date()
    @State private var displayedMonth: Date = Date()

    private let calendar = Calendar.current
    private let weekdaySymbols: [String] = {
        var cal = Calendar.current
        cal.locale = Locale(identifier: "zh_CN")
        return cal.shortWeekdaySymbols
    }()

    // MARK: - Computed

    private var completedSessions: [Session] {
        allSessions.filter { !$0.isActive }
    }

    /// 按日聚合 — key: "yyyy-MM-dd"
    private var sessionsByDay: [String: [Session]] {
        Dictionary(grouping: completedSessions) { Self.dayKey(from: $0.startedAt) }
    }

    /// 当前展示月的所有日期格子（含上月尾 + 下月头填充）
    private var calendarDays: [CalendarDay] {
        buildCalendarDays(for: displayedMonth)
    }

    /// 当月统计
    private var monthStats: MonthStats {
        let range = calendar.range(of: .day, in: .month, for: displayedMonth) ?? 1..<2
        let comps = calendar.dateComponents([.year, .month], from: displayedMonth)
        var totalSessions = 0
        var staminaSum = 0.0
        var staminaCount = 0
        var dailyAvgs: [(date: Date, avg: Double)] = []

        for day in range {
            var dc = comps
            dc.day = day
            guard let date = calendar.date(from: dc) else { continue }
            let key = Self.dayKey(from: date)
            let daySessions = sessionsByDay[key] ?? []
            totalSessions += daySessions.count
            let avgs = daySessions.compactMap(\.avgStamina)
            if !avgs.isEmpty {
                let dayAvg = avgs.reduce(0, +) / Double(avgs.count)
                staminaSum += dayAvg
                staminaCount += 1
                dailyAvgs.append((date, dayAvg))
            }
        }

        return MonthStats(
            sessionCount: totalSessions,
            avgStamina: staminaCount > 0 ? staminaSum / Double(staminaCount) : 0,
            activeDays: staminaCount,
            dailyAvgs: dailyAvgs
        )
    }

    /// 选中日的会话列表
    private var selectedDaySessions: [Session] {
        let key = Self.dayKey(from: selectedDate)
        return (sessionsByDay[key] ?? []).sorted { $0.startedAt > $1.startedAt }
    }

    /// 选中日的日统计
    private var selectedDayStats: DayStats {
        let sessions = selectedDaySessions
        let avgs = sessions.compactMap(\.avgStamina)
        let totalMin = Int(sessions.reduce(0) { $0 + $1.duration } / 60)
        return DayStats(
            sessionCount: sessions.count,
            totalMin: totalMin,
            avgStamina: avgs.isEmpty ? 0 : avgs.reduce(0, +) / Double(avgs.count)
        )
    }

    // MARK: - Body

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                // ── 第一层：月份导航 + 英雄统计 ──
                monthHeader
                    .padding(.horizontal)
                    .padding(.bottom, 16)

                // ── 第二层：日历网格 ──
                calendarGrid
                    .padding(.horizontal, 8)

                // ── 分隔 ──
                Divider()
                    .padding(.vertical, 12)

                // ── 第三层：选中日详情 ──
                selectedDayDetail
                    .padding(.horizontal)
                    .padding(.bottom, 16)
            }
        }
    }

    // MARK: - Month Header (BIOSORA 式英雄统计)

    private var monthHeader: some View {
        VStack(spacing: 12) {
            // 月份导航
            HStack {
                Button {
                    withAnimation(.easeInOut(duration: 0.3)) {
                        displayedMonth = calendar.date(byAdding: .month, value: -1, to: displayedMonth) ?? displayedMonth
                    }
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.body.weight(.semibold))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Text(displayedMonth, format: .dateTime.year().month(.wide))
                    .font(.system(size: 17, weight: .semibold))
                    .contentTransition(.numericText())

                Spacer()

                Button {
                    withAnimation(.easeInOut(duration: 0.3)) {
                        let next = calendar.date(byAdding: .month, value: 1, to: displayedMonth) ?? displayedMonth
                        if next <= Date() { displayedMonth = next }
                    }
                } label: {
                    Image(systemName: "chevron.right")
                        .font(.body.weight(.semibold))
                        .foregroundStyle(canGoForward ? .secondary : .quaternary)
                }
                .disabled(!canGoForward)
            }

            // 英雄统计行 — BIOSORA 式大数字
            HStack(alignment: .bottom, spacing: 24) {
                // 主指标：平均续航
                VStack(alignment: .leading, spacing: 2) {
                    Text(monthStats.avgStamina > 0 ? "\(Int(monthStats.avgStamina))" : "--")
                        .font(.system(size: 42, weight: .bold, design: .rounded))
                        .foregroundStyle(Flux.Colors.forStaminaValue(monthStats.avgStamina))
                        .contentTransition(.numericText())
                    Text("平均续航")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                }

                // 次要指标
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(monthStats.sessionCount)")
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)
                    Text("次专注")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("\(monthStats.activeDays)")
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)
                    Text("活跃天")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                // 月趋势 sparkline
                if monthStats.dailyAvgs.count >= 2 {
                    monthSparkline
                }
            }
        }
    }

    private var canGoForward: Bool {
        let nextMonth = calendar.date(byAdding: .month, value: 1, to: displayedMonth) ?? displayedMonth
        return nextMonth <= Date()
    }

    private var monthSparkline: some View {
        Chart(monthStats.dailyAvgs, id: \.date) { entry in
            LineMark(
                x: .value("日", entry.date),
                y: .value("续航", entry.avg)
            )
            .interpolationMethod(.catmullRom)
            .foregroundStyle(Flux.Colors.accent.opacity(0.8))
            .lineStyle(StrokeStyle(lineWidth: 1.5, lineCap: .round))

            AreaMark(
                x: .value("日", entry.date),
                yStart: .value("底", 0),
                yEnd: .value("续航", entry.avg)
            )
            .interpolationMethod(.catmullRom)
            .foregroundStyle(
                .linearGradient(
                    colors: [Flux.Colors.accent.opacity(0.15), Flux.Colors.accent.opacity(0.02)],
                    startPoint: .top, endPoint: .bottom
                )
            )
        }
        .chartYScale(domain: 0...100)
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .frame(width: 80, height: 36)
    }

    // MARK: - Calendar Grid (BIOSORA 点阵风格)

    private var calendarGrid: some View {
        let columns = Array(repeating: GridItem(.flexible(), spacing: 2), count: 7)

        return VStack(spacing: 4) {
            // 星期标头
            LazyVGrid(columns: columns, spacing: 0) {
                ForEach(weekdaySymbols, id: \.self) { symbol in
                    Text(symbol)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.tertiary)
                        .frame(height: 20)
                }
            }

            // 日期格子
            LazyVGrid(columns: columns, spacing: 4) {
                ForEach(calendarDays) { day in
                    DayCellView(
                        day: day,
                        isSelected: calendar.isDate(day.date, inSameDayAs: selectedDate),
                        isToday: calendar.isDateInToday(day.date),
                        isCurrentMonth: day.isCurrentMonth
                    )
                    .onTapGesture {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            selectedDate = day.date
                        }
                    }
                }
            }
        }
    }

    // MARK: - Selected Day Detail

    private var selectedDayDetail: some View {
        VStack(alignment: .leading, spacing: 12) {
            // 日期 + 统计摘要
            HStack(alignment: .bottom) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(selectedDate, format: .dateTime.month(.abbreviated).day().weekday(.wide))
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.primary)

                    if selectedDayStats.sessionCount > 0 {
                        HStack(spacing: 12) {
                            Label("\(selectedDayStats.sessionCount) 次", systemImage: "flame.fill")
                            Label("\(selectedDayStats.totalMin)m", systemImage: "clock")
                            if selectedDayStats.avgStamina > 0 {
                                Label("\(Int(selectedDayStats.avgStamina))", systemImage: "bolt.fill")
                                    .foregroundStyle(Flux.Colors.forStaminaValue(selectedDayStats.avgStamina))
                            }
                        }
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                // 跳到今天
                if !calendar.isDateInToday(selectedDate) {
                    Button {
                        withAnimation {
                            selectedDate = Date()
                            displayedMonth = Date()
                        }
                    } label: {
                        Text("今天")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(Flux.Colors.accent)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(Flux.Colors.accent.opacity(0.1), in: Capsule())
                    }
                }
            }

            // 会话卡片列表
            if selectedDaySessions.isEmpty {
                emptyDayView
            } else {
                ForEach(selectedDaySessions) { session in
                    NavigationLink(destination: SessionDetailView(session: session)) {
                        CalendarSessionCard(session: session)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var emptyDayView: some View {
        HStack(spacing: 10) {
            Image(systemName: "moon.stars")
                .font(.title3)
                .foregroundStyle(.quaternary)
            Text("这天没有记录")
                .font(.subheadline)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.vertical, 28)
    }

    // MARK: - Helpers

    private static func dayKey(from date: Date) -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd"
        return fmt.string(from: date)
    }

    private func buildCalendarDays(for month: Date) -> [CalendarDay] {
        let comps = calendar.dateComponents([.year, .month], from: month)
        guard let firstOfMonth = calendar.date(from: comps),
              let range = calendar.range(of: .day, in: .month, for: firstOfMonth) else { return [] }

        let firstWeekday = calendar.component(.weekday, from: firstOfMonth) // 1=Sun
        let leadingBlanks = (firstWeekday - calendar.firstWeekday + 7) % 7

        var days: [CalendarDay] = []

        // 上月填充
        if leadingBlanks > 0 {
            for offset in (1...leadingBlanks).reversed() {
                if let d = calendar.date(byAdding: .day, value: -offset, to: firstOfMonth) {
                    let key = Self.dayKey(from: d)
                    let sessions = sessionsByDay[key] ?? []
                    let avgs = sessions.compactMap(\.avgStamina)
                    days.append(CalendarDay(
                        date: d, dayNumber: calendar.component(.day, from: d),
                        isCurrentMonth: false,
                        sessionCount: sessions.count,
                        avgStamina: avgs.isEmpty ? nil : avgs.reduce(0, +) / Double(avgs.count)
                    ))
                }
            }
        }

        // 本月日期
        for day in range {
            var dc = comps
            dc.day = day
            if let d = calendar.date(from: dc) {
                let key = Self.dayKey(from: d)
                let sessions = sessionsByDay[key] ?? []
                let avgs = sessions.compactMap(\.avgStamina)
                days.append(CalendarDay(
                    date: d, dayNumber: day,
                    isCurrentMonth: true,
                    sessionCount: sessions.count,
                    avgStamina: avgs.isEmpty ? nil : avgs.reduce(0, +) / Double(avgs.count)
                ))
            }
        }

        // 补满最后一行（7的倍数）
        let trailing = (7 - days.count % 7) % 7
        if let lastDay = days.last?.date, trailing > 0 {
            for offset in 1...trailing {
                if let d = calendar.date(byAdding: .day, value: offset, to: lastDay) {
                    days.append(CalendarDay(
                        date: d, dayNumber: calendar.component(.day, from: d),
                        isCurrentMonth: false, sessionCount: 0, avgStamina: nil
                    ))
                }
            }
        }

        return days
    }
}

// MARK: - Data Models

private struct CalendarDay: Identifiable {
    let date: Date
    let dayNumber: Int
    let isCurrentMonth: Bool
    let sessionCount: Int
    let avgStamina: Double?

    var id: Date { date }
}

private struct MonthStats {
    let sessionCount: Int
    let avgStamina: Double
    let activeDays: Int
    let dailyAvgs: [(date: Date, avg: Double)]
}

private struct DayStats {
    let sessionCount: Int
    let totalMin: Int
    let avgStamina: Double
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Day Cell (BIOSORA 点阵数据格子)
// 每个格子 = 日期数字 + stamina 数据圆点
// 圆点大小 ∝ 会话数，颜色 = stamina 状态色
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

private struct DayCellView: View {
    let day: CalendarDay
    let isSelected: Bool
    let isToday: Bool
    let isCurrentMonth: Bool

    private var dotColor: Color {
        guard let avg = day.avgStamina else { return .clear }
        return Flux.Colors.forStaminaValue(avg)
    }

    private var dotSize: CGFloat {
        guard day.sessionCount > 0 else { return 0 }
        // 1 次 = 6pt, 2 次 = 8pt, 3+ 次 = 10pt
        return min(10, CGFloat(4 + day.sessionCount * 2))
    }

    var body: some View {
        VStack(spacing: 3) {
            // 日期数字
            Text("\(day.dayNumber)")
                .font(.system(size: 13, weight: isToday ? .bold : (isSelected ? .semibold : .regular),
                              design: .rounded))
                .foregroundStyle(textColor)

            // 数据圆点 — BIOSORA 核心：日历格子内的数据可视化
            if day.sessionCount > 0 {
                ZStack {
                    // 光晕底层
                    Circle()
                        .fill(dotColor.opacity(0.15))
                        .frame(width: dotSize + 4, height: dotSize + 4)
                    // 实心圆点
                    Circle()
                        .fill(dotColor)
                        .frame(width: dotSize, height: dotSize)
                }
                .frame(height: 14)
            } else {
                // 无数据占位
                Circle()
                    .fill(Color.primary.opacity(isCurrentMonth ? 0.04 : 0.02))
                    .frame(width: 4, height: 4)
                    .frame(height: 14)
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: 46)
        .background(cellBackground)
        .clipShape(.rect(cornerRadius: 8))
        .animation(.easeOut(duration: 0.15), value: isSelected)
    }

    private var textColor: Color {
        if !isCurrentMonth { return Color.secondary.opacity(0.35) }
        if isSelected { return Flux.Colors.accent }
        if isToday { return .primary }
        return .primary
    }

    @ViewBuilder
    private var cellBackground: some View {
        if isSelected {
            Flux.Colors.accent.opacity(0.08)
        } else if isToday {
            Color.primary.opacity(0.04)
        } else {
            Color.clear
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Calendar Session Card
// BIOSORA 风格：续航环 + 标题 + 标签行 + sparkline
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

private struct CalendarSessionCard: View {
    let session: Session

    private var timeRange: String {
        let fmt = DateFormatter()
        fmt.dateFormat = "HH:mm"
        let start = fmt.string(from: session.startedAt)
        let end = fmt.string(from: session.startedAt.addingTimeInterval(session.duration))
        return "\(start) – \(end)"
    }

    var body: some View {
        HStack(spacing: 12) {
            staminaRing

            VStack(alignment: .leading, spacing: 4) {
                // 第一层：时间范围
                Text(timeRange)
                    .font(.subheadline.weight(.medium))
                    .lineLimit(1)

                // 第二层：标签行
                HStack(spacing: 10) {
                    Label(Flux.formatDuration(session.duration), systemImage: "clock")
                    if let count = session.segmentCount, count > 0 {
                        Label("\(count) 段", systemImage: "rectangle.split.1x2")
                    }
                    Label(session.source.displayName, systemImage: session.source.icon)
                }
                .font(.caption2)
                .foregroundStyle(.secondary)

                // 第三层：sparkline
                if let curve = session.staminaCurveData,
                   let values = try? JSONDecoder().decode([Double].self, from: curve),
                   values.count >= 2 {
                    miniSparkline(values)
                }
            }

            Spacer()

            // 右侧：趋势指标 + 箭头
            VStack(alignment: .trailing, spacing: 4) {
                if let avg = session.avgStamina {
                    Text("\(Int(avg))")
                        .font(.system(size: 16, weight: .bold, design: .rounded))
                        .foregroundStyle(Flux.Colors.forStaminaValue(avg))
                }
                Image(systemName: "chevron.right")
                    .font(.caption2)
                    .foregroundStyle(.quaternary)
            }
        }
        .padding(14)
        .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))
    }

    // MARK: - Stamina Ring

    private var staminaRing: some View {
        let avg = session.avgStamina ?? 0
        let color = Flux.Colors.forStaminaValue(avg)

        return ZStack {
            Circle()
                .stroke(color.opacity(0.12), lineWidth: 3)
                .frame(width: 40, height: 40)
            Circle()
                .trim(from: 0, to: avg / 100)
                .stroke(color, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .frame(width: 40, height: 40)
                .rotationEffect(.degrees(-90))
            Text("\(Int(avg))")
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .foregroundStyle(color)
        }
    }

    // MARK: - Sparkline

    private func miniSparkline(_ values: [Double]) -> some View {
        let step = max(1, values.count / 30)
        let sampled = stride(from: 0, to: values.count, by: step).map { values[$0] }

        return GeometryReader { geo in
            let maxV = sampled.max() ?? 100
            let minV = sampled.min() ?? 0
            let range = max(maxV - minV, 1)

            Path { path in
                for (i, v) in sampled.enumerated() {
                    let x = geo.size.width * CGFloat(i) / CGFloat(max(sampled.count - 1, 1))
                    let y = geo.size.height * (1 - CGFloat((v - minV) / range))
                    if i == 0 { path.move(to: CGPoint(x: x, y: y)) }
                    else { path.addLine(to: CGPoint(x: x, y: y)) }
                }
            }
            .stroke(Flux.Colors.accent.opacity(0.5), lineWidth: 1.5)
        }
        .frame(height: 14)
    }
}
