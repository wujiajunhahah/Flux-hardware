"""In-memory buffers for waveform visualization and feature extraction."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List

import numpy as np

from .stream import CHANNEL_COUNT, EMGSample


class EMGRingBuffer:
    """Stores the most recent samples for plotting and RMS computation."""

    def __init__(self, max_seconds: float, sample_rate: int) -> None:
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.maxlen = int(max_seconds * sample_rate)
        self.timestamps: Deque[float] = deque(maxlen=self.maxlen)
        self.buffers: List[Deque[float]] = [deque(maxlen=self.maxlen) for _ in range(CHANNEL_COUNT)]

    def extend(self, samples: Iterable[EMGSample]) -> None:
        for sample in samples:
            self.timestamps.append(sample.timestamp)
            values = sample.values
            for idx in range(CHANNEL_COUNT):
                self.buffers[idx].append(float(values[idx]))

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.timestamps:
            return np.array([]), np.zeros((CHANNEL_COUNT, 0))
        times = np.array(self.timestamps, dtype=float)
        base = times[0]
        times = times - base
        data = np.zeros((CHANNEL_COUNT, len(times)))
        for idx in range(CHANNEL_COUNT):
            data[idx, :] = np.array(self.buffers[idx], dtype=float)
        return times, data

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
