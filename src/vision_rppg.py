"""
FluxChi -- pyVHR/rPPG adapter
=========================================================
从 vision_frame.skin.roiRgb 估计 HR + HRV。

策略:
  - 输入: 每帧 ROI RGB 均值 (0-1)
  - 主路径: 轻量频域估计（无外部依赖）
  - 扩展: 后续可替换为 pyVHR 全流程
=========================================================
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np


@dataclass
class RppgReading:
    hr_bpm: Optional[float]
    hrv_rmssd_ms: Optional[float]
    quality: float
    sample_count: int
    window_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hr_bpm": round(self.hr_bpm, 1) if self.hr_bpm is not None else None,
            "hrv_rmssd_ms": round(self.hrv_rmssd_ms, 1) if self.hrv_rmssd_ms is not None else None,
            "quality": round(self.quality, 2),
            "sample_count": self.sample_count,
            "window_sec": round(self.window_sec, 2),
        }


class RppgEngine:
    """ROI RGB -> 心率/HRV 估计器（pyVHR 兼容输入）。"""

    def __init__(self, history_sec: float = 30.0, min_hr: float = 45.0, max_hr: float = 180.0) -> None:
        self.history_sec = max(10.0, float(history_sec))
        self.min_hr = float(min_hr)
        self.max_hr = float(max_hr)
        self._rgb: Deque[Tuple[float, float, float, float]] = deque()
        self.latest: Optional[RppgReading] = None

    def update(self, frame: Dict[str, Any]) -> Optional[RppgReading]:
        if frame.get("type") != "vision_frame":
            return self.latest
        t = frame.get("t")
        if t is None:
            return self.latest
        skin = frame.get("skin") or {}
        rgb = skin.get("roiRgb") or {}
        if not isinstance(rgb, dict):
            return self.latest
        try:
            r = float(rgb.get("r"))
            g = float(rgb.get("g"))
            b = float(rgb.get("b"))
        except Exception:
            return self.latest

        self._rgb.append((float(t), r, g, b))
        self._evict(float(t))
        reading = self._estimate()
        self.latest = reading
        return reading

    def _evict(self, now: float) -> None:
        cutoff = now - self.history_sec
        while self._rgb and self._rgb[0][0] < cutoff:
            self._rgb.popleft()

    def _estimate(self) -> RppgReading:
        if len(self._rgb) < 30:
            return RppgReading(
                hr_bpm=None,
                hrv_rmssd_ms=None,
                quality=0.0,
                sample_count=len(self._rgb),
                window_sec=0.0,
            )

        arr = np.asarray(self._rgb, dtype=np.float64)
        ts = arr[:, 0]
        g = arr[:, 2]
        span = float(ts[-1] - ts[0])
        if span <= 1e-6:
            return RppgReading(None, None, 0.0, len(self._rgb), 0.0)

        fs = max(1.0, (len(ts) - 1) / span)
        sig = g - np.mean(g)
        if np.std(sig) < 1e-6:
            return RppgReading(None, None, 0.05, len(self._rgb), span)

        # 频域主峰估计心率
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
        psd = np.abs(np.fft.rfft(sig)) ** 2
        fmin = self.min_hr / 60.0
        fmax = self.max_hr / 60.0
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            return RppgReading(None, None, 0.1, len(self._rgb), span)
        band_freqs = freqs[mask]
        band_psd = psd[mask]
        idx = int(np.argmax(band_psd))
        peak_f = float(band_freqs[idx])
        hr = peak_f * 60.0

        # 粗略 HRV (从瞬时相位导数近似得到 IBI 再计算 RMSSD)
        phase = np.unwrap(np.angle(np.fft.rfft(sig, n=len(sig) * 2)))
        dphase = np.diff(phase)
        inst_f = np.clip((dphase / (2 * math.pi)) * fs, 0.1, 4.0)
        ibi_ms = 1000.0 / np.clip(inst_f, 0.4, 3.5)
        if len(ibi_ms) > 3:
            diff = np.diff(ibi_ms)
            hrv_rmssd = float(np.sqrt(np.mean(diff**2)))
        else:
            hrv_rmssd = None

        snr_like = float(np.max(band_psd) / (np.mean(band_psd) + 1e-9))
        quality = float(np.clip((snr_like - 1.0) / 10.0, 0.0, 1.0))
        return RppgReading(hr_bpm=hr, hrv_rmssd_ms=hrv_rmssd, quality=quality, sample_count=len(self._rgb), window_sec=span)

