"""Model training helpers for gesture recognition."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit

from .features import FeatureExtractor


def _load_recording(path: Path) -> np.ndarray:
    with path.open() as fh:
        reader = csv.DictReader(fh)
        rows: List[List[float]] = []
        for row in reader:
            channels = [float(row.get(f"ch{idx+1}", 0.0)) for idx in range(8)]
            rows.append(channels)
    if not rows:
        return np.zeros((0, 8))
    return np.array(rows, dtype=float)


class Trainer:
    def __init__(
        self,
        data_dir: str = "data",
        model_dir: str = "model",
        sample_rate: int = 1_000,
        window_seconds: float = 0.5,
        stride_seconds: float = 0.1,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds

    def available_recordings(self) -> Dict[str, List[Path]]:
        recordings: Dict[str, List[Path]] = {}
        if not self.data_dir.exists():
            return recordings
        for label_dir in self.data_dir.iterdir():
            if not label_dir.is_dir():
                continue
            files = sorted(p for p in label_dir.glob("*.csv"))
            if files:
                recordings[label_dir.name] = files
        return recordings

    def train(self) -> Optional[Dict[str, str]]:
        recordings = self.available_recordings()
        if not recordings:
            print("[trainer] No recordings found. Record gestures first (press R).")
            return None

        extractor = FeatureExtractor(
            sample_rate=self.sample_rate,
            window_seconds=self.window_seconds,
            stride_seconds=self.stride_seconds,
        )

        features: List[List[float]] = []
        labels: List[str] = []
        groups: List[str] = []

        for label, files in recordings.items():
            for path in files:
                data = _load_recording(path)
                if not data.size:
                    continue
                feats, _ = extractor.transform(data)
                if not feats:
                    continue
                features.extend(feats)
                labels.extend([label] * len(feats))
                groups.extend([str(path)] * len(feats))

        if len(set(labels)) < 2:
            print("[trainer] Need recordings for at least two gestures.")
            return None

        X = np.array(features)
        y = np.array(labels)
        g = np.array(groups)

        if X.size == 0:
            print("[trainer] Not enough data to train.")
            return None

        model = RandomForestClassifier(n_estimators=200, random_state=42)

        metrics: Dict[str, str] = {}
        unique_groups = np.unique(g)
        if len(unique_groups) > 1 and len(y) >= 10:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(splitter.split(X, y, g))
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            report = classification_report(y[test_idx], preds)
            metrics["report"] = report
            print(report)
        else:
            model.fit(X, y)
            metrics["report"] = "Insufficient data for hold-out validation."
            print("[trainer] Trained on full dataset (no validation split).")

        self._save_model(model, extractor)
        return metrics

    def _save_model(self, model: RandomForestClassifier, extractor: FeatureExtractor) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "model.pkl"
        config_path = self.model_dir / "config.json"
        joblib.dump(model, model_path)
        config = {
            "sample_rate": self.sample_rate,
            "window_seconds": self.window_seconds,
            "stride_seconds": self.stride_seconds,
            "feature_names": extractor.feature_names(),
        }
        with config_path.open("w") as fh:
            json.dump(config, fh, indent=2)
        print(f"[trainer] Model saved to {model_path}")


def load_model(model_dir: str = "model") -> Tuple[Optional[RandomForestClassifier], Optional[dict]]:
    model_path = Path(model_dir) / "model.pkl"
    config_path = Path(model_dir) / "config.json"
    if not model_path.exists() or not config_path.exists():
        return None, None
    model = joblib.load(model_path)
    with config_path.open() as fh:
        config = json.load(fh)
    return model, config
