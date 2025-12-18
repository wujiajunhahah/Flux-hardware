"""Gesture recording utilities."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from .stream import EMGSample


class GestureRecorder:
    def __init__(self, base_dir: str = "data") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.label: Optional[str] = None
        self._start_time: Optional[float] = None
        self._buffer: List[List[float]] = []

    def is_recording(self) -> bool:
        return self.label is not None

    def start(self, label: str) -> None:
        gesture = label.strip()
        if not gesture:
            raise ValueError("Gesture label cannot be empty")
        self.label = gesture
        self._start_time = None
        self._buffer = []
        (self.base_dir / gesture).mkdir(parents=True, exist_ok=True)
        print(f"[recorder] Recording gesture '{gesture}'. Press R again to stop.")

    def append(self, samples: Iterable[EMGSample]) -> None:
        if not self.is_recording():
            return
        for sample in samples:
            if self._start_time is None:
                self._start_time = sample.timestamp
            rel_t = sample.timestamp - self._start_time
            row = [rel_t, *sample.values.tolist()]
            self._buffer.append(row)

    def stop(self) -> Optional[Path]:
        if not self.is_recording():
            return None
        label = self.label
        self.label = None
        if not self._buffer:
            print("[recorder] No data captured.")
            return None

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        path = self.base_dir / str(label) / f"{timestamp}.csv"
        header = ["t"] + [f"ch{idx+1}" for idx in range(len(self._buffer[0]) - 1)]
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(self._buffer)
        print(f"[recorder] Saved {len(self._buffer)} samples to {path}")
        self._buffer = []
        return path
