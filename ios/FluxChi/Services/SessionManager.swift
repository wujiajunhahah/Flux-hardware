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

    /// 累计已暂停的时长（秒）；在 elapsed/duration 显示时扣除，使"已暂停"期间时间不再增长。
    private var totalPausedSec: TimeInterval = 0
    /// 当前暂停区间起点；resume 或 end 时合入 `totalPausedSec`。
    private var pausedAt: Date?

    /// 当前应展示的"实际工作时长"——已减去暂停累计。
    private func currentActiveElapsed(now: Date = Date()) -> TimeInterval {
        guard let session = activeSession else { return 0 }
        let raw = (session.endedAt ?? now).timeIntervalSince(session.startedAt)
        let live = pausedAt.map { now.timeIntervalSince($0) } ?? 0
        return max(0, raw - totalPausedSec - live)
    }

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
        totalPausedSec = 0
        pausedAt = nil

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
        guard !isPaused else { return }
        isPaused = true
        pausedAt = Date()
        snapshotTimer?.invalidate()
        snapshotTimer = nil
    }

    func resumeSession() {
        guard isPaused else { return }
        if let start = pausedAt {
            totalPausedSec += Date().timeIntervalSince(start)
        }
        pausedAt = nil
        isPaused = false
        startSnapshotTimer()
    }

    func endSession() -> Session? {
        guard let ctx = modelContext, let session = activeSession else { return nil }

        // 若结束时仍处于暂停态，先合入暂停累计
        if let start = pausedAt {
            totalPausedSec += Date().timeIntervalSince(start)
            pausedAt = nil
        }

        activeSegment?.endedAt = Date()
        session.endedAt = Date()

        stopTimers()

        let finished = session
        activeSession = nil
        activeSegment = nil
        isRecording = false
        isPaused = false
        elapsed = 0
        totalPausedSec = 0

        ctx.saveLogged()
        return finished
    }

    // MARK: - Timers

    private func startTimers() {
        startSnapshotTimer()

        elapsedTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.elapsed = self.currentActiveElapsed()
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
