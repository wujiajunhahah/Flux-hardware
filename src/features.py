"""Feature extraction helpers integrating LibEMG with manual fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .stream import CHANNEL_COUNT

try:  # pragma: no cover - optional dependency
    from libemg.feature_extractor import FeatureExtractor as LibEMGFeatureExtractor
except Exception:  # pragma: no cover
    LibEMGFeatureExtractor = None


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
    feature_group: str = "HTD"

    def __post_init__(self) -> None:
        self.window_samples = max(1, int(self.window_seconds * self.sample_rate))
        self.stride_samples = max(1, int(self.stride_seconds * self.sample_rate))
        self._pair_indices: List[Tuple[int, int]] = [
            (i, j) for i in range(CHANNEL_COUNT) for j in range(i + 1, CHANNEL_COUNT)
        ]
        self._lib_extractor = None
        self._lib_available = False
        if LibEMGFeatureExtractor is not None:
            try:
                self._lib_extractor = LibEMGFeatureExtractor()
                self._lib_available = True
            except Exception:
                self._lib_extractor = None
                self._lib_available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transform(self, data: np.ndarray) -> Tuple[List[List[float]], np.ndarray]:
        if data.ndim != 2 or data.shape[1] != CHANNEL_COUNT:
            raise ValueError("Expected data shape (samples, 8)")

        lib_features = self._lib_features(data)
        if lib_features is not None:
            centers = np.arange(len(lib_features)) * self.stride_samples + self.window_samples // 2
            return [feat.tolist() for feat in lib_features], centers

        # Manual fallback
        features: List[List[float]] = []
        centers: List[int] = []
        start = 0
        total_samples = data.shape[0]
        while start + self.window_samples <= total_samples:
            window = data[start : start + self.window_samples]
            features.append(self._fallback_features(window))
            centers.append(start + self.window_samples // 2)
            start += self.stride_samples
        return features, np.array(centers)

    def transform_window(self, window: np.ndarray) -> List[float]:
        if window.ndim != 2 or window.shape[1] != CHANNEL_COUNT:
            raise ValueError("Expected window shape (samples, 8)")
        if window.shape[0] != self.window_samples:
            raise ValueError("Window length mismatch")

        lib_features = self._lib_features(window)
        if lib_features is not None and len(lib_features):
            return lib_features[0].tolist()
        return self._fallback_features(window)

    def feature_names(self) -> List[str]:
        names: List[str] = []
        metrics = ["MAV", "RMS", "WL", "ZC", "SSC"]
        for idx in range(CHANNEL_COUNT):
            for metric in metrics:
                names.append(f"ch{idx+1}_{metric}")
        for i, j in self._pair_indices:
            names.append(f"corr_ch{i+1}_ch{j+1}")
        return names

    # ------------------------------------------------------------------
    # LibEMG integration
    # ------------------------------------------------------------------
    def _lib_features(self, data: np.ndarray) -> Optional[np.ndarray]:
        if not self._lib_available or self._lib_extractor is None:
            return None
        try:
            return self._lib_extractor.extract_features(
                data,
                window_size=self.window_samples,
                window_increment=self.stride_samples,
                sample_rate=self.sample_rate,
                feature_groups=[self.feature_group],
            )
        except Exception:
            # Disable lib integration on failure and fall back
            self._lib_available = False
            return None

    # ------------------------------------------------------------------
    # Manual fallback
    # ------------------------------------------------------------------
    def _fallback_features(self, window: np.ndarray) -> List[float]:
        feats: List[float] = []
        for idx in range(CHANNEL_COUNT):
            feats.extend(channel_features(window[:, idx], self.threshold))
        feats.extend(self._spatial_features(window))
        return feats

    def _spatial_features(self, window: np.ndarray) -> List[float]:
        if window.shape[0] < 2:
            return [0.0] * len(self._pair_indices)
        try:
            corr_matrix = np.corrcoef(window, rowvar=False)
        except Exception:
            return [0.0] * len(self._pair_indices)
        if np.isnan(corr_matrix).any():
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        features: List[float] = []
        for i, j in self._pair_indices:
            features.append(float(corr_matrix[i, j]))
        return features


def summarize_recordings(records: Dict[str, np.ndarray], extractor: FeatureExtractor) -> Tuple[List[List[float]], List[str]]:
    all_features: List[List[float]] = []
    labels: List[str] = []
    for label, values in records.items():
        feats, _ = extractor.transform(values)
        labels.extend([label] * len(feats))
        all_features.extend(feats)
    return all_features, labels
