"""Real-time inference helpers for gesture recognition."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional

import numpy as np

from .features import FeatureExtractor
from .stream import CHANNEL_COUNT, EMGSample


@dataclass
class PredictionResult:
    label: str
    confidence: float
    timestamp: float
    probabilities: Optional[List[float]] = None
    stable: bool = False


class GestureInference:
    """Consumes streaming samples and provides smoothed predictions."""

    def __init__(
        self,
        model,
        extractor: FeatureExtractor,
        stable_windows: int = 3,
        threshold: float = 0.8,
    ) -> None:
        self.model = model
        self.extractor = extractor
        self.stable_windows = max(1, stable_windows)
        self.threshold = threshold
        self.window_samples = extractor.window_samples
        self.stride_samples = extractor.stride_samples
        self._buffer: Deque[np.ndarray] = deque(maxlen=self.window_samples * 10)
        self._time_buffer: Deque[float] = deque(maxlen=self.window_samples * 10)
        self._since_stride = 0
        self._streak_label: Optional[str] = None
        self._streak_count = 0
        self._streak_reported = False
        self.latest_result: Optional[PredictionResult] = None

    def has_model(self) -> bool:
        return self.model is not None

    def reset(self) -> None:
        self._buffer.clear()
        self._time_buffer.clear()
        self._since_stride = 0
        self._streak_label = None
        self._streak_count = 0
        self._streak_reported = False
        self.latest_result = None

    def process(self, samples: Iterable[EMGSample]) -> List[PredictionResult]:
        if not self.has_model():
            return []

        results: List[PredictionResult] = []
        for sample in samples:
            self._append_sample(sample)
            result = self._maybe_predict()
            if result is not None:
                self.latest_result = result
                results.append(result)
        return results

    def _append_sample(self, sample: EMGSample) -> None:
        self._buffer.append(sample.values.astype(float))
        self._time_buffer.append(sample.timestamp)
        self._since_stride += 1

    def _maybe_predict(self) -> Optional[PredictionResult]:
        if len(self._buffer) < self.window_samples:
            return None
        if self._since_stride < self.stride_samples:
            return None
        self._since_stride -= self.stride_samples

        window = np.stack(list(self._buffer)[-self.window_samples :])
        features = self.extractor.transform_window(window)
        probabilities = self._predict_proba(features)
        best_idx = int(np.argmax(probabilities))
        label = str(self.model.classes_[best_idx])
        confidence = float(probabilities[best_idx])
        timestamp = self._time_buffer[-1]

        stable = self._update_streak(label, confidence)
        return PredictionResult(
            label=label,
            confidence=confidence,
            timestamp=timestamp,
            probabilities=list(probabilities),
            stable=stable,
        )

    def _predict_proba(self, features: List[float]) -> np.ndarray:
        prob = self.model.predict_proba([features])[0]
        return np.array(prob, dtype=float)

    def _update_streak(self, label: str, confidence: float) -> bool:
        if confidence >= self.threshold:
            if label == self._streak_label:
                self._streak_count += 1
            else:
                self._streak_label = label
                self._streak_count = 1
                self._streak_reported = False
        else:
            self._streak_label = None
            self._streak_count = 0
            self._streak_reported = False
            return False

        if self._streak_label is None:
            return False
        if self._streak_count >= self.stable_windows and not self._streak_reported:
            self._streak_reported = True
            return True
        return False
