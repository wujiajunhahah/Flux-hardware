"""Psychophysiological state estimation for Project Flux."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional

import numpy as np

from .stream import CHANNEL_COUNT, IMUSample


@dataclass
class StateReadings:
    fatigue: float
    anxiety: float
    focus: float


class StateEstimator:
    """Estimate fatigue, anxiety, and focus from EMG + IMU data."""

    def __init__(
        self,
        sample_rate: int,
        window_seconds: float = 0.5,
        fatigue_history: float = 60.0,
        imu_history: float = 5.0,
        w_tension: float = 0.6,
        w_jerk: float = 0.4,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = max(1, int(window_seconds * sample_rate))
        self.window_seconds = window_seconds
        self.fatigue_history = fatigue_history
        self.imu_history_seconds = max(window_seconds, imu_history)
        self.w_tension = w_tension
        self.w_jerk = w_jerk

        self._mnf_history: Deque[tuple[float, float]] = deque()
        self._mnf_baseline: Optional[float] = None
        self._imu_history: Deque[IMUSample] = deque()
        self._latest_state: Optional[StateReadings] = None

    def update_imu(self, samples: Iterable[IMUSample]) -> None:
        for sample in samples:
            self._imu_history.append(sample)
        self._trim_imu_history()

    def update(self, times: np.ndarray, data: np.ndarray) -> Optional[StateReadings]:
        if data.size == 0 or times.size == 0:
            return self._latest_state

        window = data[:, -self.window_samples :]
        timestamp = float(times[-1])

        mnf = self._mean_power_frequency(window)
        self._update_mnf_history(timestamp, mnf)
        fatigue = self._fatigue_score()

        cci = self._co_contraction_index(window)
        jerk = self._jerk_metric()
        anxiety = self._clamp(self.w_tension * cci + self.w_jerk * jerk)

        rhythmicity = self._rhythmicity(window)
        focus = self._clamp(rhythmicity * (1.0 - fatigue) * (1.0 - anxiety))

        self._latest_state = StateReadings(fatigue=fatigue, anxiety=anxiety, focus=focus)
        return self._latest_state

    # ------------------------------------------------------------------
    # Fatigue
    # ------------------------------------------------------------------
    def _mean_power_frequency(self, window: np.ndarray) -> float:
        if window.shape[1] < 2:
            return 0.0
        freqs = np.fft.rfftfreq(window.shape[1], d=1.0 / self.sample_rate)
        spectrum = np.abs(np.fft.rfft(window, axis=1)) ** 2
        power = np.sum(spectrum, axis=1)
        weighted = np.sum(spectrum * freqs, axis=1)
        mnf = np.divide(weighted, power, out=np.zeros_like(weighted), where=power > 0)
        return float(np.mean(mnf))

    def _update_mnf_history(self, timestamp: float, mnf: float) -> None:
        if mnf <= 0:
            return
        if self._mnf_baseline is None:
            self._mnf_baseline = mnf
        self._mnf_history.append((timestamp, mnf))
        while self._mnf_history and timestamp - self._mnf_history[0][0] > self.fatigue_history:
            self._mnf_history.popleft()

    def _fatigue_score(self) -> float:
        if not self._mnf_history or self._mnf_baseline is None:
            return 0.0
        _, latest_mnf = self._mnf_history[-1]
        drop = max(0.0, (self._mnf_baseline - latest_mnf) / (self._mnf_baseline + 1e-6))
        slope = self._mnf_slope()
        slope_term = max(0.0, -slope / (self._mnf_baseline + 1e-6))
        return self._clamp(0.6 * drop + 0.4 * slope_term)

    def _mnf_slope(self) -> float:
        if len(self._mnf_history) < 2:
            return 0.0
        times = np.array([item[0] for item in self._mnf_history], dtype=float)
        values = np.array([item[1] for item in self._mnf_history], dtype=float)
        normalized_time = times - times[0]
        if np.allclose(normalized_time, 0.0):
            return 0.0
        slope, _ = np.polyfit(normalized_time, values, 1)
        return float(slope)

    # ------------------------------------------------------------------
    # Anxiety helpers
    # ------------------------------------------------------------------
    def _co_contraction_index(self, window: np.ndarray) -> float:
        if window.size == 0:
            return 0.0
        half = CHANNEL_COUNT // 2
        flexors = np.abs(window[:half, :])
        extensors = np.abs(window[half:, :])
        flex_activity = np.mean(flexors, axis=0)
        ext_activity = np.mean(extensors, axis=0)
        numerator = np.minimum(flex_activity, ext_activity)
        denominator = np.maximum(flex_activity, ext_activity) + 1e-6
        cci = np.mean(numerator / denominator)
        return float(np.clip(cci, 0.0, 1.0))

    def _jerk_metric(self) -> float:
        if len(self._imu_history) < 2:
            return 0.0
        accels = np.array([sample.accel for sample in self._imu_history], dtype=float)
        times = np.array([sample.timestamp for sample in self._imu_history], dtype=float)
        diffs = np.diff(accels, axis=0)
        dt = np.diff(times)
        valid = dt > 1e-6
        if not np.any(valid):
            return 0.0
        jerk_vectors = diffs[valid] / dt[valid][:, None]
        jerk_magnitudes = np.linalg.norm(jerk_vectors, axis=1)
        return float(np.clip(np.sqrt(np.mean(jerk_magnitudes**2)), 0.0, 1.0))

    def _trim_imu_history(self) -> None:
        if not self._imu_history:
            return
        latest_time = self._imu_history[-1].timestamp
        while (
            self._imu_history
            and latest_time - self._imu_history[0].timestamp > self.imu_history_seconds
        ):
            self._imu_history.popleft()

    # ------------------------------------------------------------------
    # Focus / Rhythmicity
    # ------------------------------------------------------------------
    def _rhythmicity(self, window: np.ndarray) -> float:
        if window.shape[1] < 4:
            return 0.0
        signal = np.mean(window, axis=0)
        signal -= np.mean(signal)
        if np.allclose(signal, 0.0):
            return 0.0
        autocorr = np.correlate(signal, signal, mode="full")
        autocorr = autocorr[autocorr.size // 2 :]
        lag = max(1, int(self.sample_rate * 0.2))
        if lag >= autocorr.size:
            return 0.0
        rhythm = autocorr[lag] / (autocorr[0] + 1e-6)
        return float(np.clip(rhythm, 0.0, 1.0))

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))
