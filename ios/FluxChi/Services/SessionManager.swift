import SwiftUI
import SwiftData

@MainActor
final class SessionManager: ObservableObject {

    @Published var activeSession: Session?
    @Published var activeSegment: Segment?
    @Published var isRecording = false
    @Published var isPaused = false
    @Published var elapsed: TimeInterval = 0

    private var modelContext: ModelContext?
    private var stateProvider: (() -> FluxState?)?
    private var snapshotTimer: Timer?
    private var elapsedTimer: Timer?

    func configure(modelContext: ModelContext, stateProvider: @escaping () -> FluxState?) {
        self.modelContext = modelContext
        self.stateProvider = stateProvider
    }

    // MARK: - Session Lifecycle

    func startSession(source: SessionSource = .wifi) {
        guard let ctx = modelContext else { return }

        let session = Session(source: source)
        session.title = Self.generateTitle()
        ctx.insert(session)

        let segment = Segment(label: .work)
        segment.session = session
        session.segments.append(segment)
        ctx.insert(segment)

        activeSession = session
        activeSegment = segment
        isRecording = true
        isPaused = false

        startTimers()
        ctx.saveLogged()
    }

    func addSegment(label: SegmentLabel) {
        guard let ctx = modelContext, let session = activeSession else { return }

        activeSegment?.endedAt = Date()

        let segment = Segment(label: label)
        segment.session = session
        session.segments.append(segment)
        ctx.insert(segment)

        activeSegment = segment
        ctx.saveLogged()
    }

    func pauseSession() {
        isPaused = true
        snapshotTimer?.invalidate()
        snapshotTimer = nil
    }

    func resumeSession() {
        isPaused = false
        startSnapshotTimer()
    }

    func endSession() -> Session? {
        guard let ctx = modelContext, let session = activeSession else { return nil }

        activeSegment?.endedAt = Date()
        session.endedAt = Date()

        stopTimers()

        let finished = session
        activeSession = nil
        activeSegment = nil
        isRecording = false
        isPaused = false
        elapsed = 0

        ctx.saveLogged()
        return finished
    }

    // MARK: - Timers

    private func startTimers() {
        startSnapshotTimer()

        elapsedTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, let session = self.activeSession else { return }
                self.elapsed = session.duration
            }
        }
    }

    private func startSnapshotTimer() {
        let interval = TimeInterval(Flux.App.snapshotIntervalMs) / 1000.0
        snapshotTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.captureSnapshot()
            }
        }
    }

    private func stopTimers() {
        snapshotTimer?.invalidate()
        snapshotTimer = nil
        elapsedTimer?.invalidate()
        elapsedTimer = nil
    }

    private func captureSnapshot() {
        guard let ctx = modelContext,
              let segment = activeSegment,
              let state = stateProvider?(),
              !isPaused else { return }

        let snapshot = FluxSnapshot(from: state)
        snapshot.segment = segment
        segment.snapshots.append(snapshot)
        ctx.insert(snapshot)

        if segment.snapshots.count % 10 == 0 {
            ctx.saveLogged()
        }
    }

    // MARK: - Title Generation

    private static func generateTitle() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "zh_CN")
        formatter.dateFormat = "M月d日 HH:mm"
        let timeStr = formatter.string(from: Date())

        let hour = Calendar.current.component(.hour, from: Date())
        let period: String
        switch hour {
        case 5..<9:   period = "早间"
        case 9..<12:  period = "上午"
        case 12..<14: period = "午间"
        case 14..<18: period = "下午"
        case 18..<22: period = "晚间"
        default:      period = "深夜"
        }

        return "\(period)工作 · \(timeStr)"
    }
}
