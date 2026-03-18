import Foundation
import SwiftData
import SwiftUI

// MARK: - Session

@Model
final class Session {
    @Attribute(.unique) var id: UUID
    var startedAt: Date
    var endedAt: Date?
    var title: String
    var sourceRaw: String
    var schemaVersion: Int

    @Relationship(deleteRule: .cascade, inverse: \Segment.session)
    var segments: [Segment] = []

    @Relationship(deleteRule: .cascade, inverse: \UserFeedback.session)
    var feedback: UserFeedback?

    var summaryText: String?
    var avgStamina: Double?
    var minStamina: Double?
    var maxStamina: Double?
    var workDurationSec: Double?
    var restDurationSec: Double?
    var segmentCount: Int?
    var staminaCurveData: Data?

    var source: SessionSource {
        get { SessionSource(rawValue: sourceRaw) ?? .wifi }
        set { sourceRaw = newValue.rawValue }
    }

    var duration: TimeInterval {
        (endedAt ?? Date()).timeIntervalSince(startedAt)
    }

    var isActive: Bool { endedAt == nil }

    var staminaCurve: [Double] {
        guard let data = staminaCurveData else { return [] }
        return (try? JSONDecoder().decode([Double].self, from: data)) ?? []
    }

    init(title: String = "", source: SessionSource = .wifi) {
        self.id = UUID()
        self.startedAt = Date()
        self.title = title
        self.sourceRaw = source.rawValue
        self.schemaVersion = Flux.App.schemaVersion
    }
}

// MARK: - Segment

@Model
final class Segment {
    @Attribute(.unique) var id: UUID
    var labelRaw: String
    var startedAt: Date
    var endedAt: Date?

    var session: Session?

    @Relationship(deleteRule: .cascade, inverse: \FluxSnapshot.segment)
    var snapshots: [FluxSnapshot] = []

    var label: SegmentLabel {
        get { SegmentLabel(rawValue: labelRaw) ?? .work }
        set { labelRaw = newValue.rawValue }
    }

    var duration: TimeInterval {
        (endedAt ?? Date()).timeIntervalSince(startedAt)
    }

    var isActive: Bool { endedAt == nil }

    init(label: SegmentLabel = .work) {
        self.id = UUID()
        self.labelRaw = label.rawValue
        self.startedAt = Date()
    }
}

// MARK: - FluxSnapshot (captured every 500ms)

@Model
final class FluxSnapshot {
    var timestamp: Date
    var stamina: Double
    var stateRaw: String
    var consistency: Double
    var tension: Double
    var fatigue: Double
    var activity: String
    var confidence: Double
    var rms0: Double
    var rms1: Double
    var rms2: Double
    var rms3: Double
    var rms4: Double
    var rms5: Double
    var rms6: Double
    var rms7: Double

    var segment: Segment?

    var state: StaminaState {
        StaminaState(rawValue: stateRaw) ?? .focused
    }

    var rms: [Double] {
        [rms0, rms1, rms2, rms3, rms4, rms5, rms6, rms7]
    }

    init(from fluxState: FluxState) {
        self.timestamp = Date()
        self.stamina = fluxState.stamina?.value ?? 0
        self.stateRaw = fluxState.stamina?.state ?? "focused"
        self.consistency = fluxState.stamina?.consistency ?? 0
        self.tension = fluxState.stamina?.tension ?? 0
        self.fatigue = fluxState.stamina?.fatigue ?? 0
        self.activity = fluxState.activity
        self.confidence = fluxState.confidence
        let r = fluxState.rms
        self.rms0 = r.indices.contains(0) ? r[0] : 0
        self.rms1 = r.indices.contains(1) ? r[1] : 0
        self.rms2 = r.indices.contains(2) ? r[2] : 0
        self.rms3 = r.indices.contains(3) ? r[3] : 0
        self.rms4 = r.indices.contains(4) ? r[4] : 0
        self.rms5 = r.indices.contains(5) ? r[5] : 0
        self.rms6 = r.indices.contains(6) ? r[6] : 0
        self.rms7 = r.indices.contains(7) ? r[7] : 0
    }
}

// MARK: - UserFeedback

@Model
final class UserFeedback {
    @Attribute(.unique) var id: UUID
    var feelingRaw: String
    var accuracyRating: Int
    var notes: String
    var createdAt: Date

    var session: Session?

    var feeling: UserFeeling {
        get { UserFeeling(rawValue: feelingRaw) ?? .okay }
        set { feelingRaw = newValue.rawValue }
    }

    init(feeling: UserFeeling, accuracyRating: Int, notes: String = "") {
        self.id = UUID()
        self.feelingRaw = feeling.rawValue
        self.accuracyRating = accuracyRating
        self.notes = notes
        self.createdAt = Date()
    }
}

// MARK: - Enums

enum SessionSource: String, Codable, CaseIterable {
    case ble, wifi

    var displayName: String {
        switch self {
        case .ble:  return "BLE 直连"
        case .wifi: return "WiFi"
        }
    }

    var icon: String {
        switch self {
        case .ble:  return "antenna.radiowaves.left.and.right"
        case .wifi: return "wifi"
        }
    }
}

enum SegmentLabel: String, Codable, CaseIterable, Identifiable {
    case work, rest, pause, custom

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .work:   return "工作"
        case .rest:   return "休息"
        case .pause:  return "暂停"
        case .custom: return "自定义"
        }
    }

    var icon: String {
        switch self {
        case .work:   return "laptopcomputer"
        case .rest:   return "cup.and.saucer.fill"
        case .pause:  return "pause.fill"
        case .custom: return "tag.fill"
        }
    }

    var color: Color {
        switch self {
        case .work:   return Flux.Colors.accent     // 暖珊瑚
        case .rest:   return Flux.Colors.success     // 鼠尾草绿
        case .pause:  return Flux.Colors.warning     // 暖琥珀
        case .custom: return .blue
        }
    }
}

enum UserFeeling: String, Codable, CaseIterable, Identifiable {
    case focused, okay, tired, exhausted

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .focused:   return "很专注"
        case .okay:      return "还好"
        case .tired:     return "有点累"
        case .exhausted: return "很疲惫"
        }
    }

    var icon: String {
        switch self {
        case .focused:   return "bolt.fill"
        case .okay:      return "hand.thumbsup.fill"
        case .tired:     return "moon.fill"
        case .exhausted: return "zzz"
        }
    }

    var color: Color {
        switch self {
        case .focused:   return Flux.Colors.success   // 鼠尾草绿
        case .okay:      return .blue
        case .tired:     return Flux.Colors.warning    // 暖琥珀
        case .exhausted: return Flux.Colors.accent     // 暖珊瑚
        }
    }

    var staminaTarget: Double {
        switch self {
        case .focused:   return 85
        case .okay:      return 60
        case .tired:     return 35
        case .exhausted: return 10
        }
    }
}
