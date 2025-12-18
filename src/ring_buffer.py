"""In-memory buffers for waveform visualization and feature extraction."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List, Tuple

import numpy as np

from .stream import CHANNEL_COUNT, EMGSample


class EMGRingBuffer:
    """Stores the most recent samples for plotting, normalization, and RMS."""

    def __init__(self, max_seconds: float, sample_rate: int) -> None:
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.maxlen = int(max_seconds * sample_rate)
        self.timestamps: Deque[float] = deque(maxlen=self.maxlen)
        self.buffers: List[Deque[float]] = [deque(maxlen=self.maxlen) for _ in range(CHANNEL_COUNT)]

        self._window_seconds = min(2.0, max_seconds)
        self._window_len = max(1, int(self._window_seconds * sample_rate))
        self._decay = 0.99
        self._epsilon = 1e-6
        self._running_min = np.full(CHANNEL_COUNT, np.inf, dtype=float)
        self._running_max = np.full(CHANNEL_COUNT, -np.inf, dtype=float)

    def _update_scaler(self) -> None:
        if not self.timestamps:
            return
        for idx in range(CHANNEL_COUNT):
            recent = list(self.buffers[idx])[-self._window_len :]
            if not recent:
                continue
            arr = np.fromiter(recent, dtype=float)
            current_min = float(arr.min())
            current_max = float(arr.max())

            if not np.isfinite(self._running_min[idx]):
                self._running_min[idx] = current_min
            elif current_min < self._running_min[idx]:
                self._running_min[idx] = current_min
            else:
                self._running_min[idx] = (
                    self._running_min[idx] * self._decay
                    + current_min * (1.0 - self._decay)
                )

            if not np.isfinite(self._running_max[idx]):
                self._running_max[idx] = current_max
            elif current_max > self._running_max[idx]:
                self._running_max[idx] = current_max
            else:
                self._running_max[idx] = (
                    self._running_max[idx] * self._decay
                    + current_max * (1.0 - self._decay)
                )

    def extend(self, samples: Iterable[EMGSample]) -> None:
        added = False
        for sample in samples:
            self.timestamps.append(sample.timestamp)
            values = sample.values
            for idx in range(CHANNEL_COUNT):
                self.buffers[idx].append(float(values[idx]))
            added = True
        if added:
            self._update_scaler()

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.timestamps:
            return np.array([]), np.zeros((CHANNEL_COUNT, 0))
        times = np.array(self.timestamps, dtype=float)
        base = times[0]
        times = times - base
        data = np.zeros((CHANNEL_COUNT, len(times)))
        for idx in range(CHANNEL_COUNT):
            data[idx, :] = np.array(self.buffers[idx], dtype=float)
        return times, data

    def normalize(self, data: np.ndarray) -> np.ndarray:
        if data.size == 0:
            return data
        min_vals = np.where(
            np.isfinite(self._running_min), self._running_min, np.min(data, axis=1)
        )
        max_vals = np.where(
            np.isfinite(self._running_max), self._running_max, np.max(data, axis=1)
        )
        range_vals = (max_vals - min_vals + self._epsilon)[:, None]
        normalized = (data - min_vals[:, None]) / range_vals
        return np.clip(normalized, 0.0, 1.0)

    def rms(self, window_seconds: float) -> np.ndarray:
        samples = int(max(1, window_seconds * self.sample_rate))
        rms_values = np.zeros(CHANNEL_COUNT)
        for idx in range(CHANNEL_COUNT):
            buf = np.array(list(self.buffers[idx])[-samples:], dtype=float)
            if buf.size:
                rms_values[idx] = float(np.sqrt(np.mean(buf**2)))
            else:
                rms_values[idx] = 0.0
        return rms_values
