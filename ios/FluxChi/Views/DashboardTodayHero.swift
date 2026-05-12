import SwiftUI

/// Dashboard 顶部 "今日 Hero" 暗渐变大卡 —— 与分享卡 PNG 视觉对齐。
/// 把原本 4 个独立卡片（StaminaRing + Recommendation + DimensionsRow + TodaySummary）
/// 合并成单一高密度作品卡，消除"漂浮"留白。
struct DashboardTodayHero: View {
    let stamina: StaminaData?
    let decision: DecisionData?
    let ringValue: Double?
    let state: StaminaState
    let isLive: Bool
    let consistency: Double
    let tension: Double
    let fatigue: Double
    let todaySessionCount: Int
    let todayTotalMinutes: Int
    let todayAvgStamina: Double

    private var tint: Color {
        // 已连接：按 stamina 着色；未连接：用深红
        guard isLive, let v = ringValue else { return Flux.Colors.accent }
        return Flux.Colors.forStaminaValue(v)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            headerRow
            ringAndState
            Divider().overlay(Color.white.opacity(0.1))
            dimensionsStrip
            if todaySessionCount > 0 {
                Divider().overlay(Color.white.opacity(0.1))
                todayMetricStrip
            }
            if let d = decision, isLive {
                recommendationRow(d)
            }
        }
        .fluxDarkCard(tint: tint)
    }

    // MARK: - Header

    private var headerRow: some View {
        HStack {
            FluxDarkLabel(text: "今日", icon: "sparkles")
            Spacer()
            HStack(spacing: 4) {
                Circle()
                    .fill(isLive ? Color.green : Color.white.opacity(0.3))
                    .frame(width: 6, height: 6)
                    .symbolEffect(.pulse, options: .repeating, isActive: isLive)
                Text(isLive ? "实时" : "离线")
                    .font(.system(size: 10, weight: .semibold))
                    .tracking(0.8)
                    .textCase(.uppercase)
                    .foregroundStyle(.white.opacity(0.65))
            }
        }
    }

    // MARK: - Ring + State

    private var ringAndState: some View {
        HStack(alignment: .top, spacing: 20) {
            ringView
            VStack(alignment: .leading, spacing: 8) {
                Text(headline)
                    .font(.system(size: 22, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)

                Text(subline)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.7))

                if let badge = stateBadge {
                    FluxDarkChip(text: badge.text, icon: badge.icon)
                }
            }
            Spacer(minLength: 0)
        }
    }

    private var ringView: some View {
        ZStack {
            Circle()
                .stroke(Color.white.opacity(0.18), lineWidth: 7)

            if let v = ringValue {
                Circle()
                    .trim(from: 0, to: max(0.02, min(1, v / 100)))
                    .stroke(Color.white, style: StrokeStyle(lineWidth: 7, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .animation(.spring(response: 0.6, dampingFraction: 0.85), value: v)
                Text("\(Int(v))")
                    .font(.system(size: 30, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                    .contentTransition(.numericText(value: v))
            } else {
                Image(systemName: "sensor.tag.radiowaves.forward")
                    .font(.system(size: 22, weight: .regular))
                    .foregroundStyle(.white.opacity(0.5))
                    .symbolEffect(.variableColor.iterative, options: .repeating)
            }
        }
        .frame(width: 96, height: 96)
    }

    private var headline: String {
        guard isLive else { return "等待设备连接" }
        guard let v = ringValue else { return "采集信号中…" }
        if v >= 75 { return "今天身体在线" }
        if v >= 60 { return "状态在调整" }
        if v >= 40 { return "今天有起伏" }
        if v >= 20 { return "需要休息" }
        return "身体在说慢一点"
    }

    private var subline: String {
        guard isLive else { return "未连接 · 戴上手环开始" }
        guard let v = ringValue else { return "正在校准基线" }
        return state.rawValue + " · " + String(format: "%.0f / 100", v)
    }

    private struct StateBadge { let text: String; let icon: String }

    private var stateBadge: StateBadge? {
        guard isLive, let v = ringValue else { return nil }
        if v < 30 { return StateBadge(text: "续航偏低", icon: "exclamationmark.circle") }
        if v < 50 { return StateBadge(text: "建议放缓", icon: "leaf") }
        if v < 75 { return StateBadge(text: "稳态工作", icon: "target") }
        return StateBadge(text: "状态最佳", icon: "sparkle")
    }

    // MARK: - Dimensions Strip

    private var dimensionsStrip: some View {
        HStack(spacing: 0) {
            dimensionBlock(label: "一致性", value: consistency, icon: "waveform.path")
            verticalDivider
            dimensionBlock(label: "紧张度", value: tension, icon: "arrow.up.right")
            verticalDivider
            dimensionBlock(label: "疲劳度", value: fatigue, icon: "flame")
        }
    }

    private func dimensionBlock(label: String, value: Double, icon: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 9, weight: .semibold))
                Text(label)
                    .font(.system(size: 10, weight: .medium))
                    .tracking(0.5)
                    .textCase(.uppercase)
            }
            .foregroundStyle(.white.opacity(0.55))

            Text("\(Int(value * 100))")
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
                .contentTransition(.numericText(value: value))

            // mini bar
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color.white.opacity(0.15))
                        .frame(height: 3)
                    Capsule()
                        .fill(Color.white.opacity(0.85))
                        .frame(width: geo.size.width * CGFloat(max(0, min(1, value))), height: 3)
                }
            }
            .frame(height: 3)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var verticalDivider: some View {
        Rectangle()
            .fill(Color.white.opacity(0.08))
            .frame(width: 1, height: 44)
            .padding(.horizontal, 8)
    }

    // MARK: - Today Metric Strip

    private var todayMetricStrip: some View {
        HStack(spacing: 16) {
            FluxDarkMetric(
                label: "场次",
                value: "\(todaySessionCount)",
                icon: "rectangle.stack.fill"
            )
            FluxDarkMetric(
                label: "时长",
                value: todayTotalMinutes >= 60
                    ? "\(todayTotalMinutes / 60)h \(todayTotalMinutes % 60)m"
                    : "\(todayTotalMinutes)m",
                icon: "timer"
            )
            FluxDarkMetric(
                label: "平均续航",
                value: todayAvgStamina > 0 ? "\(Int(todayAvgStamina))" : "—",
                icon: "bolt.fill"
            )
        }
    }

    // MARK: - Recommendation

    private func recommendationRow(_ d: DecisionData) -> some View {
        let rec = Recommendation(rawValue: d.recommendation) ?? .keepWorking
        return HStack(alignment: .top, spacing: 10) {
            Image(systemName: rec.systemImage)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 22, height: 22)
                .background(Color.white.opacity(0.18), in: Circle())

            VStack(alignment: .leading, spacing: 2) {
                Text(rec.displayName)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.white)
                if let reason = d.reasons.first {
                    Text(reason)
                        .font(.system(size: 11))
                        .foregroundStyle(.white.opacity(0.7))
                        .lineLimit(2)
                }
            }
            Spacer(minLength: 0)
        }
    }
}
