import SwiftUI

// MARK: - Ring Engine (独立于 SwiftUI 渲染周期的信号处理引擎)

@MainActor
final class EMGRingEngine: ObservableObject {
    @Published private(set) var bars: [Double] = Array(repeating: 0, count: 8)
    @Published private(set) var energy: Double = 0

    private var currentRMS: [Double] = Array(repeating: 0, count: 8)
    private var sessionPeak: Double = 50
    private(set) var nCh: Int = 8
    private var cal: EMGCalibrationStore?
    private var timer: Timer?

    func configure(nCh: Int, calibration: EMGCalibrationStore?) {
        self.nCh = max(1, min(8, nCh))
        self.cal = calibration
    }

    func feed(_ rms: [Double]) {
        var ch = Array(rms.prefix(8))
        while ch.count < 8 { ch.append(0) }
        currentRMS = ch
    }

    func start() {
        guard timer == nil else { return }
        let t = Timer(timeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in self?.tick() }
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    /// 非对称 EMA：快攻 (α=0.45) 慢释 (α=0.12)
    private func tick() {
        let ch = currentRMS
        let slice = Array(ch.prefix(nCh))
        let maxV = slice.max() ?? 0
        sessionPeak = max(sessionPeak * 0.995, max(maxV, 25))

        var next = bars
        var totalEnergy = 0.0
        var anyChange = false

        for i in 0..<8 {
            let raw = ch[i]
            let t: Double
            if let cal, i < cal.relaxMean.count, i < cal.mvcPeak.count {
                let lo = cal.relaxMean[i]
                let hi = max(cal.mvcPeak[i], lo + 1)
                t = min(1, max(0, (raw - lo) / (hi - lo)))
            } else {
                t = min(1, raw / sessionPeak)
            }

            let prev = next[i]
            let alpha = t > prev ? 0.45 : 0.12
            let v = prev + alpha * (t - prev)
            if abs(v - prev) > 0.0005 { anyChange = true }
            next[i] = v
            totalEnergy += v
        }

        if anyChange {
            bars = next
            energy = min(1, totalEnergy / Double(nCh))
        }
    }
}

// MARK: - View

/// 径向细条圆环：条带外伸长度由各通道 RMS 驱动，非对称 EMA 平滑，信号响应式光晕。
/// 使用独立 Timer 引擎驱动 60fps 平滑插值，与 SwiftUI 渲染周期解耦。
struct FluxRadialEMGRingView: View {
    let rms: [Double]
    var activeChannelCount: Int = 8
    var barCount: Int = 128
    var theme: Theme = .light
    var calibration: EMGCalibrationStore?

    enum Theme { case light, dark }

    @StateObject private var engine = EMGRingEngine()

    private var nCh: Int { max(1, min(8, activeChannelCount)) }

    var body: some View {
        GeometryReader { geo in
            let side = min(geo.size.width, geo.size.height)
            let snapshot = engine.bars
            let eng = engine.energy

            ZStack {
                ringCanvas(side: side, opacity: 0.45, snapshot: snapshot)
                    .blur(radius: side * 0.06)
                    .scaleEffect(1.0 + 0.04 * CGFloat(eng))
                ringCanvas(side: side, opacity: 1, snapshot: snapshot)
            }
            .frame(width: side, height: side)
            .position(x: geo.size.width / 2, y: geo.size.height / 2)
        }
        .aspectRatio(1, contentMode: .fit)
        .onAppear {
            engine.configure(nCh: nCh, calibration: calibration)
            engine.feed(rms)
            engine.start()
        }
        .onDisappear { engine.stop() }
        .onChange(of: rms) { _, v in engine.feed(v) }
        .onChange(of: activeChannelCount) { _, v in
            engine.configure(nCh: max(1, min(8, v)), calibration: calibration)
        }
    }

    // MARK: - Canvas

    private func ringCanvas(side: CGFloat, opacity: Double, snapshot: [Double]) -> some View {
        let nChannels = nCh
        let count = max(barCount, nChannels * 8)
        let currentTheme = theme

        return Canvas { context, size in
            let cx = size.width / 2
            let cy = size.height / 2
            let scale = min(size.width, size.height)
            let innerR = scale * 0.26
            let outerR = scale * 0.44

            for i in 0..<count {
                let sector = min(nChannels - 1, Int((Double(i) + 0.5) / Double(count) * Double(nChannels)))
                let mag = sector < snapshot.count ? snapshot[sector] : 0

                let tLen = innerR + (outerR - innerR) * CGFloat(0.08 + 0.92 * mag)
                let theta = Double(i) / Double(count) * 2 * .pi - .pi / 2
                let ct = cos(theta), st = sin(theta)

                var path = Path()
                path.move(to: CGPoint(x: cx + CGFloat(ct) * innerR, y: cy + CGFloat(st) * innerR))
                path.addLine(to: CGPoint(x: cx + CGFloat(ct) * tLen, y: cy + CGFloat(st) * tLen))

                let hue = 0.47 + (Double(i) / Double(count)) * 0.2
                let sat: CGFloat = currentTheme == .dark ? 0.55 : 0.65
                let bri: CGFloat = currentTheme == .dark
                    ? CGFloat(0.7 + 0.3 * mag)
                    : CGFloat(0.6 + 0.3 * mag)

                context.stroke(
                    path,
                    with: .color(Color(hue: hue, saturation: Double(sat), brightness: Double(bri), opacity: opacity)),
                    style: StrokeStyle(lineWidth: max(1.2, scale / 220), lineCap: .round)
                )
            }
        }
        .drawingGroup()
    }
}

// MARK: - Previews

#Preview("Light") {
    FluxRadialEMGRingView(rms: [120, 80, 200, 90, 60, 140, 0, 0], activeChannelCount: 6, theme: .light)
        .frame(height: 280)
        .padding()
        .background(Color(.systemGroupedBackground))
}

#Preview("Dark") {
    FluxRadialEMGRingView(rms: [200, 150, 300, 100, 80, 220, 50, 40], activeChannelCount: 8, theme: .dark)
        .frame(height: 280)
        .padding()
        .background(Color.black)
}
