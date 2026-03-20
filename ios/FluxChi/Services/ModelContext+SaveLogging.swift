import SwiftData

extension ModelContext {
    /// Save with structured logging via FluxLogger.
    /// Falls back to assertionFailure in DEBUG so issues surface early.
    func saveLogged(_ label: String = #function) {
        do {
            try save()
        } catch {
            FluxLog.storage.error("SwiftData save failed [\(label)]", error: error)
            #if DEBUG
            assertionFailure("SwiftData save failed [\(label)]: \(error)")
            #endif
        }
    }
}
