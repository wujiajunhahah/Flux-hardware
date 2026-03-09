import SwiftUI
import SwiftData

struct FluxCalendarView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Session.startedAt, order: .reverse) private var allSessions: [Session]

    @State private var selectedDate: Date = Date()
    @State private var displayedMonth: Date = Date()

    private let calendar = Calendar.current
    private let weekdaySymbols = ["一", "二", "三", "四", "五", "六", "日"]

    private var completedSessions: [Session] {
        allSessions.filter { !$0.isActive }
    }

    private var sessionsByDay: [String: [Session]] {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd"
        return Dictionary(grouping: completedSessions) { fmt.string(from: $0.startedAt) }
    }

    private var selectedDaySessions: [Session] {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd"
        let key = fmt.string(from: selectedDate)
        return (sessionsByDay[key] ?? []).sorted { $0.startedAt > $1.startedAt }
    }

    private var monthTitle: String {
        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "zh_CN")
        fmt.dateFormat = "yyyy年M月"
        return fmt.string(from: displayedMonth)
    }

    var body: some View {
        VStack(spacing: 0) {
            monthHeader
            weekdayHeader
            calendarGrid
            Divider().padding(.vertical, 8)
            daySessionsList
        }
    }

    // MARK: - Month Header

    private var monthHeader: some View {
        HStack {
            Button { shiftMonth(-1) } label: {
                Image(systemName: "chevron.left")
                    .font(.caption.weight(.semibold))
            }

            Spacer()

            Text(monthTitle)
                .font(.headline)
                .contentTransition(.numericText())

            Spacer()

            Button { shiftMonth(1) } label: {
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    // MARK: - Weekday Header

    private var weekdayHeader: some View {
        HStack(spacing: 0) {
            ForEach(weekdaySymbols, id: \.self) { sym in
                Text(sym)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity)
            }
        }
        .padding(.horizontal, 8)
        .padding(.bottom, 4)
    }

    // MARK: - Calendar Grid

    private var calendarGrid: some View {
        let days = daysInMonth()
        let cols = Array(repeating: GridItem(.flexible(), spacing: 2), count: 7)

        return LazyVGrid(columns: cols, spacing: 2) {
            ForEach(days, id: \.self) { date in
                if let date {
                    dayCell(date)
                } else {
                    Color.clear.frame(height: 40)
                }
            }
        }
        .padding(.horizontal, 8)
        .animation(.easeInOut(duration: 0.2), value: displayedMonth)
    }

    private func dayCell(_ date: Date) -> some View {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd"
        let key = fmt.string(from: date)
        let daySessions = sessionsByDay[key] ?? []
        let isToday = calendar.isDateInToday(date)
        let isSelected = calendar.isDate(date, inSameDayAs: selectedDate)

        let avgStamina = daySessions.isEmpty ? nil :
            daySessions.compactMap(\.avgStamina).reduce(0, +) / max(1, Double(daySessions.compactMap(\.avgStamina).count))

        let totalWorkMin = daySessions.reduce(0.0) { $0 + ($1.workDurationSec ?? 0) } / 60

        return Button {
            withAnimation(.easeInOut(duration: 0.15)) { selectedDate = date }
        } label: {
            VStack(spacing: 2) {
                Text("\(calendar.component(.day, from: date))")
                    .font(.system(size: 14, weight: isToday ? .bold : .regular, design: .rounded))
                    .foregroundStyle(isSelected ? .white : isToday ? Flux.Colors.accent : .primary)

                if !daySessions.isEmpty {
                    completionDots(sessions: daySessions, avgStamina: avgStamina)
                } else {
                    Color.clear.frame(height: 6)
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 40)
            .background {
                if isSelected {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Flux.Colors.accent)
                } else if isToday {
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Flux.Colors.accent, lineWidth: 1)
                }
            }
        }
        .buttonStyle(.plain)
    }

    private func completionDots(sessions: [Session], avgStamina: Double?) -> some View {
        let count = min(sessions.count, 4)
        let brightness = (avgStamina ?? 50) / 100.0

        return HStack(spacing: 2) {
            ForEach(0..<count, id: \.self) { _ in
                Circle()
                    .fill(staminaColor(avgStamina ?? 50).opacity(0.5 + brightness * 0.5))
                    .frame(width: 4, height: 4)
            }
        }
        .frame(height: 6)
    }

    private func staminaColor(_ avg: Double) -> Color {
        if avg >= 70 { return .green }
        if avg >= 45 { return .orange }
        return .red
    }

    // MARK: - Day Sessions List

    private var daySessionsList: some View {
        Group {
            if selectedDaySessions.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "moon.stars")
                        .font(.title2)
                        .foregroundStyle(.tertiary)
                    Text("这天没有记录")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
            } else {
                ScrollView {
                    VStack(spacing: Flux.Spacing.item) {
                        ForEach(selectedDaySessions) { session in
                            NavigationLink(destination: SessionDetailView(session: session)) {
                                CalendarSessionCard(session: session)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal)
                }
            }
        }
    }

    // MARK: - Helpers

    private func daysInMonth() -> [Date?] {
        let comps = calendar.dateComponents([.year, .month], from: displayedMonth)
        guard let firstDay = calendar.date(from: comps),
              let range = calendar.range(of: .day, in: .month, for: firstDay) else { return [] }

        var weekday = calendar.component(.weekday, from: firstDay)
        weekday = (weekday + 5) % 7

        var days: [Date?] = Array(repeating: nil, count: weekday)

        for day in range {
            var dc = comps
            dc.day = day
            days.append(calendar.date(from: dc))
        }

        while days.count % 7 != 0 { days.append(nil) }
        return days
    }

    private func shiftMonth(_ delta: Int) {
        withAnimation(.easeInOut(duration: 0.2)) {
            displayedMonth = calendar.date(byAdding: .month, value: delta, to: displayedMonth) ?? displayedMonth
        }
    }
}

// MARK: - Calendar Session Card

private struct CalendarSessionCard: View {
    let session: Session

    var body: some View {
        HStack(spacing: 12) {
            VStack(spacing: 4) {
                let avg = session.avgStamina ?? 0
                ZStack {
                    Circle()
                        .stroke(Color.secondary.opacity(0.1), lineWidth: 3)
                        .frame(width: 40, height: 40)
                    Circle()
                        .trim(from: 0, to: avg / 100)
                        .stroke(staminaColor(avg), style: StrokeStyle(lineWidth: 3, lineCap: .round))
                        .frame(width: 40, height: 40)
                        .rotationEffect(.degrees(-90))
                    Text("\(Int(avg))")
                        .font(.system(size: 12, weight: .bold, design: .rounded))
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(session.title)
                    .font(.subheadline.weight(.medium))
                    .lineLimit(1)

                HStack(spacing: 10) {
                    Label(Flux.formatDuration(session.duration), systemImage: "clock")
                    if let count = session.segmentCount, count > 0 {
                        Label("\(count) 段", systemImage: "rectangle.split.1x2")
                    }
                    Label(session.source.displayName, systemImage: session.source.icon)
                }
                .font(.caption2)
                .foregroundStyle(.secondary)

                if let curve = session.staminaCurveData,
                   let values = try? JSONDecoder().decode([Double].self, from: curve),
                   !values.isEmpty {
                    miniSparkline(values)
                }
            }

            Spacer()

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding()
        .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))
    }

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
            .stroke(Flux.Colors.accent.opacity(0.6), lineWidth: 1.5)
        }
        .frame(height: 16)
    }

    private func staminaColor(_ avg: Double) -> Color {
        if avg >= 70 { return .green }
        if avg >= 45 { return .orange }
        return .red
    }
}
