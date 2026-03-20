import Foundation

/// Template-based fallback generators when Foundation Models are unavailable.
@available(iOS 26.0, *)
enum NLPFallbackGenerator {

    // MARK: - Session Fallback

    static func sessionSummary(_ s: NLPSessionStats, anomalies: [NLPAnomaly] = []) -> String {
        var parts: [String] = []

        // 多样化开场（每个等级 3 个变体，随机选一个）
        if s.avgStamina >= 80 {
            parts.append([
                "\(s.timeOfDay)这 \(s.totalMinutes) 分钟状态出色，平均续航 \(Int(s.avgStamina))，身体信号非常稳。",
                "这段 \(s.totalMinutes) 分钟的专注质量很高——续航 \(Int(s.avgStamina))，肌肉数据显示你进入了很好的心流状态。",
                "\(s.totalMinutes) 分钟，续航 \(Int(s.avgStamina))，\(s.timeOfDay)的这段是今天的高光。"
            ].randomElement()!)
        } else if s.avgStamina >= 60 {
            parts.append([
                "\(s.timeOfDay)这 \(s.totalMinutes) 分钟表现不错，续航 \(Int(s.avgStamina))，身体在配合你的节奏。",
                "\(s.totalMinutes) 分钟的专注，续航 \(Int(s.avgStamina))，整体稳定，有进一步提升的空间。",
                "这段专注 \(s.totalMinutes) 分钟，续航 \(Int(s.avgStamina))——不算惊艳但很扎实。"
            ].randomElement()!)
        } else if s.avgStamina >= 40 {
            parts.append([
                "\(s.timeOfDay)这 \(s.totalMinutes) 分钟，续航 \(Int(s.avgStamina))，有波动但坚持下来了。",
                "\(s.totalMinutes) 分钟专注，续航 \(Int(s.avgStamina))，身体信号说还有提升空间——关键在休息节奏。",
                "续航 \(Int(s.avgStamina))，不是最佳状态，但\(s.totalMinutes) 分钟的坚持本身就值得认可。"
            ].randomElement()!)
        } else {
            parts.append([
                "\(s.timeOfDay)这段身体信号偏弱，续航 \(Int(s.avgStamina))，肌肉数据在反复发出疲劳信号。",
                "续航只有 \(Int(s.avgStamina))，身体今天确实在告诉你需要慢下来。",
                "这 \(s.totalMinutes) 分钟续航 \(Int(s.avgStamina))，身体需要更多恢复，不要勉强。"
            ].randomElement()!)
        }

        // 黄金窗口洞察
        if s.peakFocusMinutes >= 10 {
            parts.append("连续高效了 \(Int(s.peakFocusMinutes)) 分钟，这是你的「黄金窗口」——下次可以用它来规划番茄钟时长。")
        } else if s.peakFocusMinutes >= 5 {
            parts.append("最长高效 \(Int(s.peakFocusMinutes)) 分钟，可以试着逐步延长到 20 分钟以上。")
        }

        // 下降点洞察
        if let dp = s.declinePointMinutes, dp > 3 {
            parts.append([
                "大约第 \(dp) 分钟开始走下坡，下次在这之前主动休息 5 分钟。",
                "第 \(dp) 分钟后续航开始下降——这是你的「转折点」，记住它。"
            ].randomElement()!)
        }

        // 异常或维度分析（每次只选一个最相关的）
        let critical = anomalies.filter { $0.severity == .critical }
        let warnings = anomalies.filter { $0.severity == .warning }

        if let first = critical.first {
            parts.append(first.message + "，需要重点注意。")
        } else if let first = warnings.first {
            parts.append(first.message + "。")
        } else if s.fatigueAvg > 0.6 {
            parts.append([
                "疲劳指数 \(Int(s.fatigueAvg * 100))% 偏高，站起来活动一下，做几次深呼吸。",
                "身体疲劳度达到 \(Int(s.fatigueAvg * 100))%，喝杯水、伸个懒腰再继续。"
            ].randomElement()!)
        } else if s.tensionAvg > 0.4 {
            parts.append("紧张度 \(Int(s.tensionAvg * 100))% 偏高，检查一下坐姿——屏幕是否在视线正前方、肩膀有没有耸起来。")
        } else if s.consistencyAvg > 0.7 {
            parts.append("动作一致性 \(Int(s.consistencyAvg * 100))% 非常稳定，说明你的工作姿势很到位。")
        } else if s.staminaDelta > 10 {
            parts.append("续航整体呈上升趋势（+\(Int(s.staminaDelta))），越做越进入状态。")
        } else if s.staminaDelta < -15 {
            parts.append("续航下降了 \(Int(abs(s.staminaDelta))) 点，下次可以缩短时长、增加休息频率。")
        }

        return parts.joined(separator: "")
    }

    // MARK: - Daily Insight Fallback

    static func dailyInsight(_ s: NLPDailyStats, anomalies: [NLPAnomaly] = []) -> String {
        let hour = Calendar.current.component(.hour, from: Date())
        let greeting = hour < 12 ? "早上好" : hour < 18 ? "下午好" : "辛苦了"

        // 多样化开场（基于数据特征选择不同角度）
        var candidates: [String] = []

        if s.avgStamina >= 75 {
            candidates = [
                "\(greeting)，今天 \(s.sessionCount) 段专注、续航 \(Int(s.avgStamina))，身体状态很在线。",
                "今天节奏掌握得很好，\(s.totalMinutes) 分钟专注、平均续航 \(Int(s.avgStamina))，身体信号很稳。",
                "今天身体给了很正的反馈，\(s.sessionCount) 段专注续航都在高位，保持这个感觉。"
            ]
        } else if s.avgStamina >= 50 {
            candidates = [
                "\(greeting)，今天 \(s.sessionCount) 段专注共 \(s.totalMinutes) 分钟，续航 \(Int(s.avgStamina))，有波动但整体还行。",
                "今天的身体数据中规中矩（续航 \(Int(s.avgStamina))），\(s.trend == "上升" ? "好消息是越往后状态越好" : "试试在疲劳前就主动休息")。",
                "\(s.totalMinutes) 分钟的专注不算少，续航 \(Int(s.avgStamina)) 说明还有提升空间——关键在休息节奏。"
            ]
        } else {
            candidates = [
                "\(greeting)，今天身体信号偏弱，续航只有 \(Int(s.avgStamina))。没关系，状态有起伏是正常的。",
                "今天不是最佳状态（续航 \(Int(s.avgStamina))），身体可能需要更多恢复时间。",
                "续航 \(Int(s.avgStamina)) 偏低，身体在告诉你需要减速。早点休息，明天再来。"
            ]
        }

        var result = candidates.randomElement()!

        // 追加具体洞察（优先级：异常 > 趋势 > 维度分析）
        let critical = anomalies.filter { $0.severity == .critical }
        let warnings = anomalies.filter { $0.severity == .warning }

        if let first = critical.first {
            result += first.message + "，建议重点关注。"
        } else if let first = warnings.first {
            result += first.message + "。"
        } else if s.trend == "下降" && s.sessionCount >= 2 {
            let tips = [
                "后半段明显下滑，明天试试把重要任务放在上午。",
                "状态越往后越疲，下次在感觉还行的时候就休息一下。"
            ]
            result += tips.randomElement()!
        } else if s.avgFatigue > 0.5 {
            result += "疲劳指数 \(Int(s.avgFatigue * 100))% 偏高，睡前拉伸 10 分钟会有帮助。"
        } else if s.avgTension > 0.4 {
            result += "紧张度 \(Int(s.avgTension * 100))% 偏高，注意肩颈放松和坐姿调整。"
        } else if s.trend == "上升" {
            result += "而且越到后面状态越好，说明你找到了自己的节奏。"
        } else if s.totalPeakFocusMinutes >= 20 {
            result += "高效时段累计 \(Int(s.totalPeakFocusMinutes)) 分钟，质量不错。"
        } else if let bestSlot = s.timeDistribution.max(by: { $0.value < $1.value })?.key {
            result += "\(bestSlot)是你今天的最佳时段，可以把重要事情安排在这个时间。"
        }

        return result
    }

    // MARK: - Empty Day Insight

    static func emptyDayInsight(recentSessions: [Session] = []) -> String {
        let hour = Calendar.current.component(.hour, from: Date())

        if !recentSessions.isEmpty {
            let avgVals = recentSessions.compactMap(\.avgStamina)
            let recentAvg = avgVals.isEmpty ? 0 : Int(avgVals.reduce(0, +) / Double(avgVals.count))
            let totalDays = Set(recentSessions.map { Calendar.current.startOfDay(for: $0.startedAt) }).count

            if hour < 12 {
                return "早上好！最近 \(totalDays) 天平均续航 \(recentAvg)，\(recentAvg >= 65 ? "状态不错，今天继续保持" : "今天试试每 25 分钟休息一次")。连上设备开始第一段专注吧。"
            } else if hour < 18 {
                return "下午好，今天还没有记录。你最近 \(totalDays) 天的数据显示\(recentAvg >= 65 ? "状态稳定" : "还有提升空间")，找个时间段专注一下？"
            } else {
                return "今天还没有专注记录。\(recentAvg >= 65 ? "你最近状态不错，休息一晚明天继续" : "没关系，适当休息也是提升的一部分")。"
            }
        }

        // 完全没有历史数据
        if hour < 12 {
            return "早上好！连上设备开始你的第一段专注记录，教练会根据你的身体数据给出个性化建议。"
        } else if hour < 18 {
            return "开始第一段专注记录吧，教练需要你的身体数据才能给出有针对性的建议。"
        } else {
            return "晚上好，还没有专注记录。明天一早连上设备，让教练帮你找到最佳工作节奏。"
        }
    }

    // MARK: - Follow-Up Fallback

    static func followUp(question: String, stats: NLPDailyStats?, anomalies: [NLPAnomaly]) -> String {
        let q = question.lowercased()

        if q.contains("下午") || q.contains("下降") || q.contains("掉") {
            if let s = stats, s.avgFatigue > 0.5 {
                return "你的疲劳指数 \(Int(s.avgFatigue * 100))% 偏高，下午下降可能是午餐后血糖波动加上前半天的累积疲劳。两个建议：1) 午餐少吃精碳水（白米饭、面条），多吃蛋白质；2) 饭后散步 10 分钟再开始工作。"
            }
            if let s = stats, s.avgTension > 0.4 {
                return "你的紧张度偏高（\(Int(s.avgTension * 100))%），下午下降可能和持续的肩颈紧绷有关。试试每小时做一次 2 分钟的肩部放松——耸肩 5 秒、放下，重复 5 次。"
            }
            return "下午续航下降是最常见的模式，主要原因是午餐后的血糖波动和上午的疲劳累积。最有效的对策：午饭后走 10 分钟，下午第一个番茄钟从 15 分钟开始，逐渐加到 25 分钟。"
        }

        if q.contains("延长") || q.contains("高效") || q.contains("时间") || q.contains("更长") {
            if let s = stats, s.totalPeakFocusMinutes > 0 {
                let current = Int(s.totalPeakFocusMinutes)
                let target = min(current + 10, 45)
                return "你目前最长高效时段 \(current) 分钟。要延长到 \(target) 分钟，关键不是「忍着不休息」，而是在第 \(max(current - 5, 15)) 分钟时主动站起来活动 2 分钟——这种微休息能帮你把高效状态延续更久。"
            }
            return "延长高效时间的核心原则是「主动休息」——在状态还好的时候就短暂休息 3-5 分钟，比连续工作到筋疲力尽再休息效果好得多。试试 25 分钟工作 + 5 分钟休息的节奏。"
        }

        if q.contains("紧张") || q.contains("tension") || q.contains("肩") || q.contains("颈") {
            let hasAnomaly = anomalies.contains { $0.type == .highTension }
            if hasAnomaly {
                return "你的紧张度确实偏高，这说明肩颈区域在长时间紧绷。三个立刻能做的事：1) 把屏幕抬高到视线平齐；2) 每 30 分钟做一次「肩部画圈」——向前转 5 圈、向后转 5 圈；3) 打字时注意放松双肩，不要耸起来。"
            }
            if let s = stats {
                let level = s.avgTension > 0.5 ? "偏高" : s.avgTension > 0.3 ? "正常偏上" : "正常范围"
                return "你的紧张度 \(Int(s.avgTension * 100))%，属于\(level)。紧张度主要反映肩颈肌肉的持续收缩程度。低于 30% 是理想状态，30-50% 需要注意姿势，超过 50% 建议立刻做放松练习。"
            }
            return "适度紧张是正常的工作状态，但持续超过 40% 就需要注意了。最简单的判断方法：如果你需要刻意才能放松肩膀，说明紧张度已经偏高。每隔 30 分钟检查一次肩膀是否耸起来。"
        }

        if q.contains("疲劳") || q.contains("累") || q.contains("fatigue") {
            if let s = stats {
                if s.avgFatigue > 0.6 {
                    return "你的疲劳指数 \(Int(s.avgFatigue * 100))% 确实偏高。疲劳不是一下子出现的，而是逐渐累积的。建议今天早点结束工作，睡前做 10 分钟拉伸。明天试试缩短每段专注的时长（20 分钟），增加休息次数。"
                } else {
                    return "疲劳指数 \(Int(s.avgFatigue * 100))% 还在可控范围。要保持低疲劳，关键是不要等到累了才休息——在感觉还行的时候就主动休息 5 分钟，这样全天的疲劳累积会明显减少。"
                }
            }
            return "疲劳的核心管理原则是「预防大于治疗」。与其等到精疲力尽再休息 30 分钟，不如每 25 分钟就休息 5 分钟。后者全天总工作量反而更高。"
        }

        if q.contains("恢复") || q.contains("休息") || q.contains("recovery") {
            let hasPoorRecovery = anomalies.contains { $0.type == .poorRecovery }
            if hasPoorRecovery {
                return "你的恢复效率确实不太够——休息后续航回升不明显。可能原因：1) 休息时间太短（至少需要 5-10 分钟）；2) 休息时还在看手机（大脑没有真正休息）。真正的休息是：站起来、看远处、做几次深呼吸。"
            }
            return "有效的休息不是刷手机，而是让身体和大脑都切换模式。最高效的恢复方式：站起来走一圈、看 20 秒远处的绿色植物、做 5 次深呼吸。这比坐在椅子上刷 15 分钟手机恢复效果好 3 倍。"
        }

        if q.contains("最佳") || q.contains("什么时候") || q.contains("时段") || q.contains("时间段") {
            if let s = stats, let bestSlot = s.timeDistribution.max(by: { $0.value < $1.value })?.key {
                return "从今天的数据看，\(bestSlot)是你状态最好的时段。一般来说，大多数人的认知高峰在上午 9-11 点和下午 3-5 点。建议把最重要、最需要创造力的任务放在你的高峰时段。"
            }
            return "每个人的最佳时段不同，但大部分人有两个高峰：上午 9-11 点和下午 3-5 点。连续记录一周后，教练就能帮你找到属于你的最佳时段。"
        }

        // 通用回答——也要有变化
        let genericAnswers = [
            "这是个好问题。从你的数据来看，建议关注两件事：1) 主动休息——每 25 分钟站起来活动 5 分钟；2) 注意姿势——肩膀放松、屏幕与视线平齐。这两个习惯坚持一周就能看到续航提升。",
            "基于你的身体数据，最重要的建议是建立固定节奏——工作 25 分钟、休息 5 分钟、每 2 小时一次长休息 15 分钟。节奏感比「坚持更久」更能提升效率。",
            "好问题。如果只能给一个建议，那就是「主动休息」——不要等身体发出疲劳信号才休息，在状态还好的时候就主动停下来 5 分钟。这一个习惯就能带来明显改变。"
        ]
        return genericAnswers.randomElement()!
    }
}
