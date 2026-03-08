"""
FluxChi — 工作续航引擎 (Stamina Engine)
=========================================================
从 EMG 信号中提取三个维度，合成一个"工作续航"分数 (0-100):

  D1: 模式一致性 (Pattern Consistency)
      EMG 激活模式的变异系数 → 专注度代理
  D2: 基线张力 (Baseline Tension)
      休息时的残余肌电 → 紧张-放松
  D3: 累积退化 (Sustained Capacity)
      MDF 频谱下降趋势 → 持续输出能力

stamina = w1*consistency + w2*(1-tension) + w3*(1-fatigue)

状态: focused (>60) → fading (30-60) → depleted (<30) → recovering
=========================================================
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

from .fatigue import FatigueEstimator, FatigueReading
from .stream import CHANNEL_COUNT

WORK_ACTIVITIES = {
    "typing", "mouse_use",
    "finger_movement", "wrist_flex", "wrist_extend", "wrist_movement",
}
REST_ACTIVITIES = {"idle", "stretching", "rest"}


class StaminaState(str, Enum):
    FOCUSED = "focused"
    FADING = "fading"
    DEPLETED = "depleted"
    RECOVERING = "recovering"


@dataclass
class StaminaReading:
    stamina: float            # 0-100
    state: StaminaState
    consistency: float        # 0-1, D1
    tension: float            # 0-1, D2 (higher = more tense)
    fatigue: float            # 0-1, D3
    drain_rate: float         # stamina units lost per real minute
    recovery_rate: float      # stamina units gained per real minute
    suggested_work_min: float
    suggested_break_min: float
    continuous_work_min: float
    total_work_min: float
    timestamp: float


class StaminaEngine:
    """Accumulative stamina engine driven by three EMG dimensions."""

    def __init__(
        self,
        sample_rate: int = 1000,
        window_sec: float = 0.25,
        consistency_window_sec: float = 30.0,
        tension_window_sec: float = 60.0,
        w1: float = 0.40,
        w2: float = 0.25,
        w3: float = 0.35,
        base_drain_per_min: float = 1.8,
        base_recovery_per_min: float = 5.0,
        speed: float = 1.0,
        smoothing_window: int = 5,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = int(window_sec * sample_rate)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.base_drain = base_drain_per_min
        self.base_recovery = base_recovery_per_min
        self.speed = speed
        self.smoothing_window = smoothing_window

        # D1: consistency tracking
        self._rms_history: Deque[Tuple[float, float]] = deque()
        self._consistency_window = consistency_window_sec

        # D2: tension tracking
        self._rms_recent: Deque[float] = deque()
        self._tension_window = tension_window_sec
        self._resting_baseline: Optional[float] = None
        self._baseline_samples: List[float] = []
        self._baseline_count = 30

        # D3: fatigue (delegates to FatigueEstimator)
        self._fatigue = FatigueEstimator(
            sample_rate=sample_rate,
            window_seconds=window_sec,
            history_seconds=120.0,
        )

        # Accumulated state
        self._stamina: float = 100.0
        self._last_timestamp: Optional[float] = None
        self._work_start: Optional[float] = None
        self._rest_start: Optional[float] = None
        self._total_work_sec: float = 0.0
        self._activity_history: List[str] = []
        self._is_working: bool = False
        self._latest: Optional[StaminaReading] = None

        # Smoothed drain/recovery rate for display
        self._recent_drain: float = self.base_drain
        self._recent_recovery: float = self.base_recovery

    def update(
        self,
        emg_window: np.ndarray,
        activity: str,
        timestamp: Optional[float] = None,
    ) -> StaminaReading:
        now = timestamp or time.time()

        if emg_window.ndim == 2 and emg_window.shape[0] == CHANNEL_COUNT and emg_window.shape[1] != CHANNEL_COUNT:
            emg_window = emg_window.T
        n_ch = min(emg_window.shape[1], CHANNEL_COUNT) if emg_window.ndim == 2 else 1

        # --- Compute RMS across channels ---
        rms_per_ch = np.sqrt(np.mean(emg_window ** 2, axis=0))
        mean_rms = float(np.mean(rms_per_ch[:n_ch]))

        # --- Activity smoothing ---
        self._activity_history.append(activity)
        if len(self._activity_history) > self.smoothing_window:
            self._activity_history = self._activity_history[-self.smoothing_window:]
        from collections import Counter
        smoothed = Counter(self._activity_history).most_common(1)[0][0]
        is_working = smoothed in WORK_ACTIVITIES

        # --- Track work/rest transitions ---
        if is_working and not self._is_working:
            self._is_working = True
            self._work_start = now
            self._rest_start = None
        elif not is_working and self._is_working:
            if self._work_start is not None:
                self._total_work_sec += now - self._work_start
            self._is_working = False
            self._rest_start = now
            self._work_start = None

        continuous_work_sec = (now - self._work_start) if self._work_start else 0.0
        total_work_sec = self._total_work_sec + (continuous_work_sec if self._work_start else 0)

        # === D1: Pattern Consistency ===
        self._rms_history.append((now, mean_rms))
        while self._rms_history and now - self._rms_history[0][0] > self._consistency_window:
            self._rms_history.popleft()

        consistency = self._compute_consistency(is_working)

        # === D2: Baseline Tension ===
        self._rms_recent.append(mean_rms)
        max_tension_samples = int(self._tension_window / 0.2)
        while len(self._rms_recent) > max_tension_samples:
            self._rms_recent.popleft()

        if not is_working:
            self._baseline_samples.append(mean_rms)
            if self._resting_baseline is None and len(self._baseline_samples) >= self._baseline_count:
                self._resting_baseline = float(np.median(self._baseline_samples))

        tension = self._compute_tension()

        # === D3: Fatigue (MDF slope) ===
        fatigue_reading = self._fatigue.update(now, emg_window)
        fatigue = fatigue_reading.score

        # === Composite instantaneous quality ===
        instant_quality = (
            self.w1 * consistency
            + self.w2 * (1.0 - tension)
            + self.w3 * (1.0 - fatigue)
        )
        instant_quality = float(np.clip(instant_quality, 0.0, 1.0))

        # === Accumulate stamina ===
        dt_real = 0.0
        if self._last_timestamp is not None:
            dt_real = max(0.0, now - self._last_timestamp)
        self._last_timestamp = now

        dt_sim = dt_real * self.speed
        dt_min = dt_sim / 60.0

        if is_working:
            fatigue_multiplier = 0.5 + fatigue
            drain = self.base_drain * fatigue_multiplier * dt_min
            low_consistency_penalty = max(0.0, 0.5 - consistency) * 0.5 * dt_min
            drain += low_consistency_penalty
            self._stamina = max(0.0, self._stamina - drain)
            self._recent_drain = 0.9 * self._recent_drain + 0.1 * (drain / max(dt_min, 1e-6))
        else:
            rest_quality = max(0.1, 1.0 - tension)
            recovery = self.base_recovery * rest_quality * dt_min
            self._stamina = min(100.0, self._stamina + recovery)
            self._recent_recovery = 0.9 * self._recent_recovery + 0.1 * (recovery / max(dt_min, 1e-6))

        # === State machine ===
        state = self._compute_state(is_working)

        # === Suggested durations ===
        drain_rate = max(0.01, self._recent_drain)
        recovery_rate = max(0.01, self._recent_recovery)

        if is_working:
            remaining = max(0.0, self._stamina - 30.0)
            suggested_work = remaining / drain_rate
            suggested_break = (70.0 - min(self._stamina, 70.0)) / recovery_rate
        else:
            deficit = 70.0 - self._stamina
            suggested_break = max(0.0, deficit / recovery_rate) if deficit > 0 else 0.0
            suggested_work = max(0.0, (self._stamina - 30.0) / drain_rate)

        self._latest = StaminaReading(
            stamina=round(self._stamina, 1),
            state=state,
            consistency=round(consistency, 3),
            tension=round(tension, 3),
            fatigue=round(fatigue, 3),
            drain_rate=round(drain_rate, 2),
            recovery_rate=round(recovery_rate, 2),
            suggested_work_min=round(suggested_work, 1),
            suggested_break_min=round(suggested_break, 1),
            continuous_work_min=round(continuous_work_sec / 60.0, 1),
            total_work_min=round(total_work_sec / 60.0, 1),
            timestamp=now,
        )
        return self._latest

    @property
    def stamina(self) -> float:
        return self._stamina

    @property
    def latest(self) -> Optional[StaminaReading]:
        return self._latest

    def reset(self) -> None:
        self._stamina = 100.0
        self._last_timestamp = None
        self._work_start = None
        self._rest_start = None
        self._total_work_sec = 0.0
        self._rms_history.clear()
        self._rms_recent.clear()
        self._activity_history.clear()
        self._fatigue.reset()
        self._resting_baseline = None
        self._baseline_samples.clear()
        self._is_working = False

    # --- Persistence ---

    def save(self, path: str = "profiles/default/state.json") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stamina": self._stamina,
            "total_work_sec": self._total_work_sec,
            "resting_baseline": self._resting_baseline,
            "saved_at": time.time(),
            "w1": self.w1, "w2": self.w2, "w3": self.w3,
            "base_drain": self.base_drain,
            "base_recovery": self.base_recovery,
        }
        p.write_text(json.dumps(data, indent=2))

    def load(self, path: str = "profiles/default/state.json") -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            saved_at = data.get("saved_at", 0)
            away_sec = time.time() - saved_at

            self._stamina = data.get("stamina", 100.0)
            self._total_work_sec = data.get("total_work_sec", 0.0)
            self._resting_baseline = data.get("resting_baseline")

            if away_sec > 600:
                away_min = away_sec / 60.0
                recovery = min(away_min * self.base_recovery * 0.3, 100.0 - self._stamina)
                self._stamina = min(100.0, self._stamina + recovery)

            return True
        except Exception:
            return False

    # --- Internal computations ---

    def _compute_consistency(self, is_working: bool) -> float:
        if not is_working or len(self._rms_history) < 5:
            return 0.0

        values = np.array([v for _, v in self._rms_history])
        mean_val = np.mean(values)
        if mean_val < 1e-6:
            return 0.0
        cv = float(np.std(values) / mean_val)
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    def _compute_tension(self) -> float:
        if len(self._rms_recent) < 5:
            return 0.0
        vals = np.array(self._rms_recent)
        p10 = float(np.percentile(vals, 10))
        baseline = self._resting_baseline or p10
        if baseline < 1e-6:
            return 0.0
        ratio = p10 / baseline
        return float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))

    def _compute_state(self, is_working: bool) -> StaminaState:
        if not is_working:
            return StaminaState.RECOVERING
        if self._stamina > 60:
            return StaminaState.FOCUSED
        if self._stamina > 30:
            return StaminaState.FADING
        return StaminaState.DEPLETED
