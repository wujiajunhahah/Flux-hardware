import SwiftUI
import SwiftData

struct FluxCalendarView: View {
    @Query(sort: \Session.startedAt, order: .reverse) private var allSessions: [Session]

    @State private var selectedDate: Date = Date()

    private static let dayKeyFormatter: DateFormatter = {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd"
        return fmt
    }()

    private var completedSessions: [Session] {
        allSessions.filter { !$0.isActive }
    }

    private var sessionsByDay: [String: [Session]] {
        Dictionary(grouping: completedSessions) { Self.dayKeyFormatter.string(from: $0.startedAt) }
    }

    private var selectedDaySessions: [Session] {
        let key = Self.dayKeyFormatter.string(from: selectedDate)
        return (sessionsByDay[key] ?? []).sorted { $0.startedAt > $1.startedAt }
    }

    // MARK: - Body

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                DatePicker(
                    "选择日期",
                    selection: $selectedDate,
                    in: ...Date(),
                    displayedComponents: .date
                )
                .datePickerStyle(.graphical)
                .tint(Flux.Colors.accent)
                .padding(.horizontal, 8)

                Divider()
                    .padding(.top, 4)

                daySessionsList
                    .padding(.top, 12)
            }
        }
    }

    // MARK: - Day Sessions List

    @ViewBuilder
    private var daySessionsList: some View {
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
            .padding(.vertical, 32)
        } else {
            VStack(spacing: Flux.Spacing.item) {
                ForEach(selectedDaySessions) { session in
                    NavigationLink(destination: SessionDetailView(session: session)) {
                        CalendarSessionCard(session: session)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 16)
        }
    }
}

// MARK: - Calendar Session Card

private struct CalendarSessionCard: View {
    let session: Session

    var body: some View {
        HStack(spacing: 12) {
            staminaRing

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

    private var staminaRing: some View {
        let avg = session.avgStamina ?? 0
        let color = Flux.Colors.forStaminaValue(avg)

        return ZStack {
            Circle()
                .stroke(color.opacity(0.15), lineWidth: 3)
                .frame(width: 40, height: 40)
            Circle()
                .trim(from: 0, to: avg / 100)
                .stroke(color, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .frame(width: 40, height: 40)
                .rotationEffect(.degrees(-90))
            Text("\(Int(avg))")
                .font(.system(size: 12, weight: .bold, design: .rounded))
        }
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
}
