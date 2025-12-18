"""Feature extraction helpers for EMG gesture modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .stream import CHANNEL_COUNT


def estimate_contact_quality(rms_value: float) -> str:
    if rms_value < 20:
        return "WEAK"
    if rms_value < 200:
        return "GOOD"
    return "NOISY"


def _zero_crossings(values: np.ndarray, threshold: float) -> int:
    count = 0
    for i in range(1, len(values)):
        prev, curr = values[i - 1], values[i]
        if abs(prev - curr) < threshold:
            continue
        if prev == 0:
            continue
        if prev > 0 >= curr or prev < 0 <= curr:
            count += 1
    return count


def _slope_changes(values: np.ndarray, threshold: float) -> int:
    count = 0
    for i in range(1, len(values) - 1):
        diff1 = values[i] - values[i - 1]
        diff2 = values[i + 1] - values[i]
        if abs(diff1) < threshold or abs(diff2) < threshold:
            continue
        if diff1 * diff2 < 0:
            count += 1
    return count


def channel_features(values: np.ndarray, threshold: float = 20.0) -> List[float]:
    mav = float(np.mean(np.abs(values))) if values.size else 0.0
    rms = float(np.sqrt(np.mean(values**2))) if values.size else 0.0
    wl = float(np.sum(np.abs(np.diff(values)))) if values.size else 0.0
    zc = float(_zero_crossings(values, threshold))
    ssc = float(_slope_changes(values, threshold))
    return [mav, rms, wl, zc, ssc]


@dataclass
class FeatureExtractor:
    sample_rate: int
    window_seconds: float = 0.5
    stride_seconds: float = 0.1
    threshold: float = 20.0

    def __post_init__(self) -> None:
        self.window_samples = max(1, int(self.window_seconds * self.sample_rate))
        self.stride_samples = max(1, int(self.stride_seconds * self.sample_rate))

    def transform(self, data: np.ndarray) -> Tuple[List[List[float]], np.ndarray]:
        """Return windowed feature vectors and window midpoints."""

        if data.ndim != 2 or data.shape[1] != CHANNEL_COUNT:
            raise ValueError("Expected data shape (samples, 8)")

        features: List[List[float]] = []
        centers: List[int] = []
        start = 0
        total_samples = data.shape[0]
        while start + self.window_samples <= total_samples:
            window = data[start : start + self.window_samples]
            features.append(self._features_for_window(window))
            centers.append(start + self.window_samples // 2)
            start += self.stride_samples
        return features, np.array(centers)

    def feature_names(self) -> List[str]:
        names = []
        metrics = ["MAV", "RMS", "WL", "ZC", "SSC"]
        for idx in range(CHANNEL_COUNT):
            for metric in metrics:
                names.append(f"ch{idx+1}_{metric}")
        return names

    def _features_for_window(self, window: np.ndarray) -> List[float]:
        if window.shape[0] != self.window_samples or window.shape[1] != CHANNEL_COUNT:
            raise ValueError("Window shape does not match configured window size")
        feats: List[float] = []
        for idx in range(CHANNEL_COUNT):
            feats.extend(channel_features(window[:, idx], self.threshold))
        return feats

    def transform_window(self, window: np.ndarray) -> List[float]:
        """Return the feature vector for a single window."""

        if window.ndim != 2:
            raise ValueError("Expected 2D array for window")
        return self._features_for_window(window)


def summarize_recordings(records: Dict[str, np.ndarray], extractor: FeatureExtractor) -> Tuple[List[List[float]], List[str]]:
    """Convert gesture→data map into features and labels."""

    all_features: List[List[float]] = []
    labels: List[str] = []
    for label, values in records.items():
        feats, _ = extractor.transform(values)
        labels.extend([label] * len(feats))
        all_features.extend(feats)
    return all_features, labels
