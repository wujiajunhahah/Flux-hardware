"""
FluxChi — 决策引擎
=========================================================
基于 StaminaEngine 的续航分数生成人类可读的建议。

输入: StaminaReading (stamina 0-100, state, dimensions)
输出: DecisionOutput (recommendation, urgency, reasons)
=========================================================
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .energy import StaminaReading, StaminaState

WORK_ACTIVITIES = {
    "typing", "mouse_use",
    "finger_movement", "wrist_flex", "wrist_extend", "wrist_movement",
}
REST_ACTIVITIES = {"idle", "stretching", "rest"}


class Recommendation(str, Enum):
    KEEP_WORKING = "keep_working"
    TAKE_BREAK = "take_break"
    START_WORKING = "start_working"
    REST_MORE = "rest_more"


@dataclass
class DecisionOutput:
    state: str
    recommendation: Recommendation
    urgency: float
    reasons: List[str]
    stamina: float
    continuous_work_min: float
    total_work_min: float
    activity: str
    suggested_work_min: float = 0.0
    suggested_break_min: float = 0.0
    timestamp: float = field(default_factory=time.time)


class DecisionEngine:
    """Generates recommendations from StaminaReading."""

    def update(self, reading: StaminaReading, activity: str) -> DecisionOutput:
        reasons: List[str] = []
        urgency = 0.0
        stamina = reading.stamina
        state = reading.state
        is_working = state in (StaminaState.FOCUSED, StaminaState.FADING, StaminaState.DEPLETED) and activity not in REST_ACTIVITIES

        if is_working:
            if stamina > 60:
                urgency = 0.0
                reasons.append(f"状态良好，预计还能专注 ~{reading.suggested_work_min:.0f} 分钟")
            elif stamina > 30:
                urgency = 0.3 + 0.4 * (1.0 - (stamina - 30) / 30.0)
                reasons.append(f"效率在下降（续航 {stamina:.0f}%），建议 {reading.suggested_work_min:.0f} 分钟内休息")
                if reading.consistency < 0.4:
                    reasons.append("工作模式不稳定，专注度下降")
                if reading.tension > 0.5:
                    reasons.append("肌肉紧张度偏高")
            else:
                urgency = 0.9
                reasons.append(f"续航耗尽（{stamina:.0f}%），需要休息")
                reasons.append(f"建议休息 ~{reading.suggested_break_min:.0f} 分钟")

            recommendation = Recommendation.TAKE_BREAK if urgency >= 0.5 else Recommendation.KEEP_WORKING
        else:
            if stamina >= 70:
                urgency = 0.0
                recommendation = Recommendation.START_WORKING
                reasons.append(f"精力恢复（{stamina:.0f}%），可以开始工作")
            elif stamina >= 50:
                urgency = 0.1
                recommendation = Recommendation.START_WORKING
                reasons.append(f"基本恢复（{stamina:.0f}%），可以开始")
            else:
                urgency = 0.0
                recommendation = Recommendation.REST_MORE
                reasons.append(f"正在恢复（{stamina:.0f}%），还需 ~{reading.suggested_break_min:.0f} 分钟")
                if reading.tension > 0.3:
                    reasons.append("身体还未完全放松")

        if not reasons:
            reasons.append("状态正常")

        return DecisionOutput(
            state=state.value,
            recommendation=recommendation,
            urgency=urgency,
            reasons=reasons,
            stamina=stamina,
            continuous_work_min=reading.continuous_work_min,
            total_work_min=reading.total_work_min,
            activity=activity,
            suggested_work_min=reading.suggested_work_min,
            suggested_break_min=reading.suggested_break_min,
            timestamp=reading.timestamp,
        )
