"""
FluxChi — 决策引擎
=========================================================
融合三层信号，输出工作/休息建议：

Layer 1: 活动分类 (RandomForest → typing / mouse_use / idle / stretching)
Layer 2: 疲劳估计 (MDF slope → fatigue_score 0-1)
Layer 3: 时间规则 (连续工作时长、一天累计时长、时段)

输出：
  - state: "working" | "resting" | "transition"
  - recommendation: "keep_working" | "take_break" | "start_working" | "rest_more"
  - urgency: 0-1
  - reason: 人类可读的原因字符串
=========================================================
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class UserState(str, Enum):
    WORKING = "working"
    RESTING = "resting"
    TRANSITION = "transition"


class Recommendation(str, Enum):
    KEEP_WORKING = "keep_working"
    TAKE_BREAK = "take_break"
    START_WORKING = "start_working"
    REST_MORE = "rest_more"


WORK_ACTIVITIES = {
    "typing", "mouse_use",
    "finger_movement", "wrist_flex", "wrist_extend", "wrist_movement",
}
REST_ACTIVITIES = {"idle", "stretching", "rest"}


@dataclass
class DecisionOutput:
    state: UserState
    recommendation: Recommendation
    urgency: float
    reasons: List[str]
    fatigue_score: float
    continuous_work_min: float
    total_work_min: float
    activity: str
    timestamp: float = field(default_factory=time.time)


class DecisionEngine:
    """Rule-based decision engine fusing activity, fatigue, and temporal context."""

    def __init__(
        self,
        max_continuous_work_min: float = 45.0,
        max_daily_work_hours: float = 8.0,
        min_break_min: float = 5.0,
        fatigue_threshold_warn: float = 0.4,
        fatigue_threshold_urgent: float = 0.7,
        peak_hours: tuple = (9, 12, 14, 17),
        smoothing_window: int = 5,
    ) -> None:
        self.max_continuous_work_min = max_continuous_work_min
        self.max_daily_work_hours = max_daily_work_hours
        self.min_break_min = min_break_min
        self.fatigue_threshold_warn = fatigue_threshold_warn
        self.fatigue_threshold_urgent = fatigue_threshold_urgent
        self.peak_hours = peak_hours
        self.smoothing_window = smoothing_window

        self._work_start: Optional[float] = None
        self._rest_start: Optional[float] = None
        self._total_work_sec: float = 0.0
        self._session_start: float = time.time()
        self._activity_history: List[str] = []
        self._current_state = UserState.RESTING
        self._latest: Optional[DecisionOutput] = None

    def update(
        self,
        activity: str,
        fatigue_score: float,
        timestamp: Optional[float] = None,
    ) -> DecisionOutput:
        """Process new classification + fatigue input, return decision."""
        now = timestamp or time.time()

        self._activity_history.append(activity)
        if len(self._activity_history) > self.smoothing_window:
            self._activity_history = self._activity_history[-self.smoothing_window:]

        smoothed_activity = self._smoothed_activity()
        is_working = smoothed_activity in WORK_ACTIVITIES
        is_resting = smoothed_activity in REST_ACTIVITIES

        if is_working and self._current_state != UserState.WORKING:
            self._current_state = UserState.WORKING
            self._work_start = now
            self._rest_start = None
        elif is_resting and self._current_state != UserState.RESTING:
            if self._work_start is not None:
                self._total_work_sec += now - self._work_start
            self._current_state = UserState.RESTING
            self._rest_start = now
            self._work_start = None

        continuous_work_sec = 0.0
        if self._work_start is not None:
            continuous_work_sec = now - self._work_start

        continuous_work_min = continuous_work_sec / 60.0
        total_work_sec = self._total_work_sec + (
            continuous_work_sec if self._work_start else 0
        )
        total_work_min = total_work_sec / 60.0
        total_work_hours = total_work_min / 60.0

        rest_duration_min = 0.0
        if self._rest_start is not None:
            rest_duration_min = (now - self._rest_start) / 60.0

        reasons: List[str] = []
        urgency = 0.0

        if is_working:
            if fatigue_score >= self.fatigue_threshold_urgent:
                urgency = max(urgency, 0.9)
                reasons.append(f"肌肉疲劳严重 ({fatigue_score:.0%})")
            elif fatigue_score >= self.fatigue_threshold_warn:
                urgency = max(urgency, 0.5)
                reasons.append(f"肌肉开始疲劳 ({fatigue_score:.0%})")

            time_ratio = continuous_work_min / self.max_continuous_work_min
            if time_ratio >= 1.0:
                urgency = max(urgency, 0.8)
                reasons.append(f"连续工作 {continuous_work_min:.0f} 分钟（超过 {self.max_continuous_work_min:.0f} 分钟上限）")
            elif time_ratio >= 0.8:
                urgency = max(urgency, 0.4)
                reasons.append(f"已连续工作 {continuous_work_min:.0f} 分钟")

            if total_work_hours >= self.max_daily_work_hours:
                urgency = max(urgency, 0.7)
                reasons.append(f"今日累计工作 {total_work_hours:.1f} 小时")

            hour = time.localtime(now).tm_hour
            is_peak = any(
                s <= hour < e
                for s, e in zip(self.peak_hours[::2], self.peak_hours[1::2])
            )
            if not is_peak and continuous_work_min > 20:
                urgency = max(urgency, 0.3)
                reasons.append("当前非高效时段")

            recommendation = (
                Recommendation.TAKE_BREAK if urgency >= 0.5
                else Recommendation.KEEP_WORKING
            )

        else:
            rested_enough = rest_duration_min >= self.min_break_min
            recovered = fatigue_score < self.fatigue_threshold_warn

            if rested_enough and recovered:
                recommendation = Recommendation.START_WORKING
                urgency = 0.1
                reasons.append(f"已休息 {rest_duration_min:.0f} 分钟，疲劳恢复")
            elif rested_enough and not recovered:
                recommendation = Recommendation.REST_MORE
                urgency = 0.3
                reasons.append(f"虽已休息 {rest_duration_min:.0f} 分钟，但疲劳未完全恢复")
            else:
                recommendation = Recommendation.REST_MORE
                urgency = 0.0
                reasons.append(f"休息中（{rest_duration_min:.0f}/{self.min_break_min:.0f} 分钟）")

        if not reasons:
            reasons.append("状态正常")

        output = DecisionOutput(
            state=self._current_state,
            recommendation=recommendation,
            urgency=urgency,
            reasons=reasons,
            fatigue_score=fatigue_score,
            continuous_work_min=continuous_work_min,
            total_work_min=total_work_min,
            activity=smoothed_activity,
            timestamp=now,
        )
        self._latest = output
        return output

    def reset_daily(self) -> None:
        self._total_work_sec = 0.0
        self._session_start = time.time()

    @property
    def latest(self) -> Optional[DecisionOutput]:
        return self._latest

    def _smoothed_activity(self) -> str:
        """Majority vote over recent activity window."""
        if not self._activity_history:
            return "idle"
        from collections import Counter
        counter = Counter(self._activity_history)
        return counter.most_common(1)[0][0]
