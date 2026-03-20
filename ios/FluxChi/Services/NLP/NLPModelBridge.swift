import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

/// Foundation Models bridge — all `#if canImport(FoundationModels)` confined here.
@available(iOS 26.0, *)
enum NLPModelBridge {

    /// Whether the on-device model is available
    static var isAvailable: Bool {
        #if canImport(FoundationModels)
        return SystemLanguageModel.default.isAvailable
        #else
        return false
        #endif
    }

    /// Diagnostic info for debugging Foundation Models degradation
    static var diagnosticInfo: String {
        #if canImport(FoundationModels)
        let model = SystemLanguageModel.default
        if model.isAvailable {
            return "Foundation Models 可用 ✓"
        } else {
            return "Foundation Models 不可用 — 请检查: 1) 设备是否支持 Apple Intelligence  2) 设置 > Apple Intelligence 是否已开启  3) 模型是否已下载完成"
        }
        #else
        return "FoundationModels 框架未编入 — 需要 Xcode 26+ 编译"
        #endif
    }

    /// Prewarm the model (call at app launch)
    static func prewarm() {
        #if canImport(FoundationModels)
        Task {
            let session = LanguageModelSession()
            session.prewarm()
            print("[NLP] Foundation Models prewarm 已请求")
        }
        #endif
    }

    /// Try generating a response via Foundation Models. Returns nil on failure/unavailability.
    static func tryGenerate(prompt: String, persona: NLPPersona = .sessionCoach) async -> String? {
        #if canImport(FoundationModels)
        let model = SystemLanguageModel.default
        guard model.isAvailable else {
            print("[NLP] Foundation Models 不可用 — isAvailable=false")
            print("[NLP] 诊断: \(diagnosticInfo)")
            return nil
        }

        do {
            let inst = persona.instructions
            let session = LanguageModelSession {
                inst
            }
            print("[NLP] Foundation Models 开始生成 (persona: \(persona))...")
            let response = try await session.respond(to: prompt)
            let text = response.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty {
                print("[NLP] Foundation Models 返回空内容，降级到模板")
                return nil
            }
            print("[NLP] Foundation Models 生成成功 (\(text.count) 字符)")
            return text
        } catch {
            print("[NLP] Foundation Models 错误: \(error.localizedDescription)")
            print("[NLP] 错误详情: \(error)")
            return nil
        }
        #else
        print("[NLP] FoundationModels 框架未编入 (canImport 失败)")
        return nil
        #endif
    }
}
