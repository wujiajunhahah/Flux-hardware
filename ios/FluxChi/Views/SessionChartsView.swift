import SwiftUI
import Charts

// MARK: - ADHD Focus Dot Map

struct ADHDDotMap: View {
    let snapshots: [FluxSnapshot]
    let sessionStart: Date

    var body: some View {
        VStack(alignment: .leading, spacing: Flux.Spacing.item) {
            FluxSectionLabel(title: "专注力热图", icon: "circle.grid.3x3.fill")

            let cols = 10
            let rows = max(1, (snapshots.count + cols - 1) / cols)
            let totalSlots = rows * cols

            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 3), count: cols), spacing: 3) {
                ForEach(0..<totalSlots, id: \.self) { i in
                    if i < snapshots.count {
                        let snap = snapshots[i]
                        let normalized = snap.stamina / 100.0
                        let stateColor = stateColor(snap.stateRaw)

                        Circle()
                            .fill(stateColor.opacity(0.3 + normalized * 0.7))
                            .frame(width: dotSize(normalized), height: dotSize(normalized))
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        Circle()
                            .fill(Color.clear)
                            .frame(width: 4, height: 4)
                    }
                }
            }
            .frame(height: CGFloat(rows) * 18)
            .padding()
            .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))

            HStack(spacing: 16) {
                dotLegend(color: Flux.Colors.forStaminaState(.focused), label: "专注")
                dotLegend(color: Flux.Colors.forStaminaState(.fading), label: "下降")
                dotLegend(color: Flux.Colors.forStaminaState(.depleted), label: "耗尽")
                dotLegend(color: Flux.Colors.forStaminaState(.recovering), label: "恢复")
            }
            .font(.caption2)
        }
    }

    private func dotSize(_ normalized: Double) -> CGFloat {
        let base: CGFloat = 6
        let maxExtra: CGFloat = 10
        return base + CGFloat(normalized) * maxExtra
    }

    private func stateColor(_ raw: String) -> Color {
        guard let state = StaminaState(rawValue: raw) else { return .gray }
        return Flux.Colors.forStaminaState(state)
    }

    private func dotLegend(color: Color, label: String) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label).foregroundStyle(.secondary)
        }
    }
}

// MARK: - Clock / Radar Chart

struct FocusClockView: View {
    let snapshots: [FluxSnapshot]
    let sessionStart: Date

    private struct SliceData: Identifiable {
        let id = UUID()
        let minuteMark: Int
        let avgStamina: Double
        let state: String
    }

    private var slices: [SliceData] {
        guard let last = snapshots.last else { return [] }
        let totalSec = last.timestamp.timeIntervalSince(sessionStart)
        let bucketCount = max(12, min(24, Int(totalSec / 60) * 2))
        let bucketSec = max(1, totalSec / Double(bucketCount))

        var result: [SliceData] = []
        for i in 0..<bucketCount {
            let start = sessionStart.addingTimeInterval(Double(i) * bucketSec)
            let end = sessionStart.addingTimeInterval(Double(i + 1) * bucketSec)
            let bucket = snapshots.filter { $0.timestamp >= start && $0.timestamp < end }
            let avg = bucket.isEmpty ? 50 : bucket.map(\.stamina).reduce(0, +) / Double(bucket.count)
            let dominantState = bucket.isEmpty ? "focused" :
                Dictionary(grouping: bucket, by: \.stateRaw)
                    .max(by: { $0.value.count < $1.value.count })?.key ?? "focused"
            result.append(SliceData(minuteMark: Int(Double(i) * bucketSec / 60), avgStamina: avg, state: dominantState))
        }
        return result
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Flux.Spacing.item) {
            FluxSectionLabel(title: "时间轮盘", icon: "clock.fill")

            GeometryReader { geo in
                let center = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                let radius = min(geo.size.width, geo.size.height) / 2 - 30

                ZStack {
                    ForEach(0..<4) { i in
                        Circle()
                            .stroke(Color.secondary.opacity(0.1), lineWidth: 0.5)
                            .frame(width: radius * 2 * CGFloat(i + 1) / 4,
                                   height: radius * 2 * CGFloat(i + 1) / 4)
                    }

                    ForEach(Array(slices.enumerated()), id: \.element.id) { idx, slice in
                        let angle = Angle.degrees(Double(idx) / Double(slices.count) * 360 - 90)
                        let dist = radius * CGFloat(slice.avgStamina / 100.0)

                        let x = center.x + dist * CGFloat(cos(angle.radians))
                        let y = center.y + dist * CGFloat(sin(angle.radians))

                        Circle()
                            .fill(stateColor(slice.state))
                            .frame(width: 8, height: 8)
                            .position(x: x, y: y)

                        if idx > 0 {
                            let prevAngle = Angle.degrees(Double(idx - 1) / Double(slices.count) * 360 - 90)
                            let prevDist = radius * CGFloat(slices[idx - 1].avgStamina / 100.0)
                            let px = center.x + prevDist * CGFloat(cos(prevAngle.radians))
                            let py = center.y + prevDist * CGFloat(sin(prevAngle.radians))

                            Path { path in
                                path.move(to: CGPoint(x: px, y: py))
                                path.addLine(to: CGPoint(x: x, y: y))
                            }
                            .stroke(stateColor(slice.state).opacity(0.4), lineWidth: 1.5)
                        }
                    }

                    VStack(spacing: 2) {
                        let totalMin = snapshots.last.map {
                            Int($0.timestamp.timeIntervalSince(sessionStart) / 60)
                        } ?? 0
                        Text("\(totalMin)")
                            .font(.system(size: 22, weight: .bold, design: .rounded))
                        Text("分钟")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .position(center)

                    ForEach([0, 3, 6, 9], id: \.self) { mark in
                        let angle = Angle.degrees(Double(mark) / 12.0 * 360 - 90)
                        let labelDist = radius + 16
                        let lx = center.x + labelDist * CGFloat(cos(angle.radians))
                        let ly = center.y + labelDist * CGFloat(sin(angle.radians))
                        let perMark = snapshots.last.map {
                            max(1, Int($0.timestamp.timeIntervalSince(sessionStart) / 60 / 12))
                        } ?? 1
                        Text(mark == 0 ? "开始" : "+\(mark * perMark)m")
                            .font(.system(size: 9, weight: .medium, design: .monospaced))
                            .foregroundStyle(.tertiary)
                            .position(x: lx, y: ly)
                    }
                }
            }
            .frame(height: 200)
            .padding()
            .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))
        }
    }

    private func stateColor(_ raw: String) -> Color {
        guard let state = StaminaState(rawValue: raw) else { return .gray }
        return Flux.Colors.forStaminaState(state)
    }
}

// MARK: - Dimension Radar

struct DimensionRadarView: View {
    let consistency: Double
    let tension: Double
    let fatigue: Double

    private let labels = ["一致性", "紧张度", "疲劳度"]

    var body: some View {
        VStack(alignment: .leading, spacing: Flux.Spacing.item) {
            FluxSectionLabel(title: "三维诊断", icon: "hexagon")

            GeometryReader { geo in
                let center = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                let radius = min(geo.size.width, geo.size.height) / 2 - 24
                let values = [consistency, tension, fatigue]

                ZStack {
                    ForEach(1...3, id: \.self) { ring in
                        polygonPath(center: center, radius: radius * CGFloat(ring) / 3, sides: 3)
                            .stroke(Color.secondary.opacity(0.15), lineWidth: 0.5)
                    }

                    ForEach(0..<3) { i in
                        let angle = Angle.degrees(Double(i) / 3.0 * 360 - 90)
                        let x = center.x + (radius + 14) * CGFloat(cos(angle.radians))
                        let y = center.y + (radius + 14) * CGFloat(sin(angle.radians))

                        Text(labels[i])
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(.secondary)
                            .position(x: x, y: y)
                    }

                    dataPolygon(center: center, radius: radius, values: values)
                        .fill(Flux.Colors.accent.opacity(0.2))

                    dataPolygon(center: center, radius: radius, values: values)
                        .stroke(Flux.Colors.accent, lineWidth: 2)

                    ForEach(0..<3) { i in
                        let angle = Angle.degrees(Double(i) / 3.0 * 360 - 90)
                        let dist = radius * CGFloat(values[i])
                        let x = center.x + dist * CGFloat(cos(angle.radians))
                        let y = center.y + dist * CGFloat(sin(angle.radians))

                        Circle()
                            .fill(Flux.Colors.accent)
                            .frame(width: 6, height: 6)
                            .position(x: x, y: y)
                    }
                }
            }
            .frame(height: 180)
            .padding()
            .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))
        }
    }

    private func polygonPath(center: CGPoint, radius: CGFloat, sides: Int) -> Path {
        Path { path in
            for i in 0...sides {
                let angle = Angle.degrees(Double(i) / Double(sides) * 360 - 90)
                let pt = CGPoint(
                    x: center.x + radius * CGFloat(cos(angle.radians)),
                    y: center.y + radius * CGFloat(sin(angle.radians))
                )
                if i == 0 { path.move(to: pt) }
                else { path.addLine(to: pt) }
            }
            path.closeSubpath()
        }
    }

    private func dataPolygon(center: CGPoint, radius: CGFloat, values: [Double]) -> Path {
        Path { path in
            for i in 0...values.count {
                let idx = i % values.count
                let angle = Angle.degrees(Double(idx) / Double(values.count) * 360 - 90)
                let dist = radius * CGFloat(values[idx])
                let pt = CGPoint(
                    x: center.x + dist * CGFloat(cos(angle.radians)),
                    y: center.y + dist * CGFloat(sin(angle.radians))
                )
                if i == 0 { path.move(to: pt) }
                else { path.addLine(to: pt) }
            }
            path.closeSubpath()
        }
    }
}

// MARK: - Enhanced Stamina Curve with gradient fill

struct StaminaCurveChart: View {
    let snapshots: [FluxSnapshot]

    var body: some View {
        VStack(alignment: .leading, spacing: Flux.Spacing.item) {
            FluxSectionLabel(title: "Stamina 曲线", icon: "chart.xyaxis.line")

            if !snapshots.isEmpty {
                Chart {
                    ForEach(Array(snapshots.enumerated()), id: \.offset) { _, snap in
                        AreaMark(
                            x: .value("时间", snap.timestamp),
                            yStart: .value("底", 0),
                            yEnd: .value("Stamina", snap.stamina)
                        )
                        .interpolationMethod(.catmullRom)
                        .foregroundStyle(
                            .linearGradient(
                                colors: [Flux.Colors.accent.opacity(0.3), Flux.Colors.accent.opacity(0.05)],
                                startPoint: .top, endPoint: .bottom
                            )
                        )

                        LineMark(
                            x: .value("时间", snap.timestamp),
                            y: .value("Stamina", snap.stamina)
                        )
                        .interpolationMethod(.catmullRom)
                        .foregroundStyle(Flux.Colors.accent)
                        .lineStyle(StrokeStyle(lineWidth: 2))
                    }

                    RuleMark(y: .value("高效", 60))
                        .foregroundStyle(Flux.Colors.success.opacity(0.3))
                        .lineStyle(StrokeStyle(dash: [4]))
                        .annotation(position: .leading) {
                            Text("60").font(.system(size: 8)).foregroundStyle(Flux.Colors.success.opacity(0.5))
                        }

                    RuleMark(y: .value("警告", 30))
                        .foregroundStyle(Flux.Colors.warning.opacity(0.3))
                        .lineStyle(StrokeStyle(dash: [4]))
                        .annotation(position: .leading) {
                            Text("30").font(.system(size: 8)).foregroundStyle(Flux.Colors.warning.opacity(0.5))
                        }
                }
                .chartYScale(domain: 0...100)
                .chartYAxis {
                    AxisMarks(values: [0, 25, 50, 75, 100]) { _ in
                        AxisValueLabel()
                            .font(.system(size: 9, design: .monospaced))
                    }
                }
                .chartXAxis {
                    AxisMarks(values: .automatic(desiredCount: 4)) { _ in
                        AxisValueLabel(format: .dateTime.minute().second())
                            .font(.system(size: 9, design: .monospaced))
                    }
                }
                .frame(height: 180)
            }
        }
        .padding()
        .background(.ultraThinMaterial, in: .rect(cornerRadius: Flux.Radius.medium))
    }
}
