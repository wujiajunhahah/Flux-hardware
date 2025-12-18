"""Matplotlib-based visualization for the EMG gesture tool."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .stream import CHANNEL_COUNT


class EMGPlotUI:
    def __init__(
        self,
        window_seconds: float,
        interval_ms: int = 50,
    ) -> None:
        plt.style.use("default")
        self.window_seconds = window_seconds
        self.interval_ms = interval_ms
        self.fig = plt.figure(figsize=(12, 8))
        gs = self.fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.3], hspace=0.35)
        self.ax_wave = self.fig.add_subplot(gs[0])
        self.ax_rms = self.fig.add_subplot(gs[1])
        self.ax_status = self.fig.add_subplot(gs[2])
        self.ax_status.axis("off")

        self.lines = [self.ax_wave.plot([], [], lw=1)[0] for _ in range(CHANNEL_COUNT)]
        self.ax_wave.set_ylabel("EMG (a.u.)")
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_title("Waveletech EMG - Real-time")

        x = np.arange(CHANNEL_COUNT)
        self.bar_rects = self.ax_rms.bar(x, np.zeros(CHANNEL_COUNT), color="royalblue")
        self.ax_rms.set_xticks(x)
        self.ax_rms.set_xticklabels([f"Ch {i+1}" for i in range(CHANNEL_COUNT)])
        self.ax_rms.set_ylabel("RMS (uV)")
        self.ax_rms.set_ylim(0, 300)

        self.quality_texts = [self.ax_rms.text(idx, 0, "", ha="center", va="bottom", fontsize=9) for idx in x]

        self.status_text = self.ax_status.text(0.01, 0.9, "", transform=self.ax_status.transAxes, fontsize=10)
        self.recording_text = self.ax_status.text(0.01, 0.5, "", transform=self.ax_status.transAxes, fontsize=10)
        self.inference_text = self.ax_status.text(0.01, 0.1, "", transform=self.ax_status.transAxes, fontsize=10)
        self.prediction_text = self.ax_wave.text(0.99, 0.95, "", transform=self.ax_wave.transAxes, ha="right", fontsize=13, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        self.anim: FuncAnimation | None = None
        self._last_render = time.time()

    def connect_keypress(self, callback) -> None:
        self.fig.canvas.mpl_connect("key_press_event", callback)

    def start_animation(self, update_func) -> None:
        self.anim = FuncAnimation(
            self.fig,
            update_func,
            interval=self.interval_ms,
            cache_frame_data=False,
        )

    def show(self) -> None:
        plt.show()

    def update_waveforms(self, times: np.ndarray, data: np.ndarray) -> None:
        if times.size == 0 or data.size == 0:
            for line in self.lines:
                line.set_data([], [])
            self.ax_wave.set_xlim(0, self.window_seconds)
            return

        max_time = max(times[-1], self.window_seconds)
        self.ax_wave.set_xlim(max(0, max_time - self.window_seconds), max_time)

        peak = float(np.max(np.abs(data))) if data.size else 1.0
        spacing = max(peak * 0.6, 100.0)
        offsets = np.arange(CHANNEL_COUNT) * spacing
        shifted = data + offsets[:, None]

        for idx, line in enumerate(self.lines):
            line.set_data(times, shifted[idx])

        self.ax_wave.set_ylim(-spacing, offsets[-1] + spacing)
        self.ax_wave.set_yticks(offsets)
        self.ax_wave.set_yticklabels([f"Ch {idx+1}" for idx in range(CHANNEL_COUNT)])

    def update_rms(self, rms_values: Sequence[float], quality_labels: Sequence[str]) -> None:
        values = np.asarray(rms_values, dtype=float)
        if values.size == 0:
            values = np.zeros(len(self.bar_rects))
        peak = max(50.0, float(values.max(initial=0.0)))
        self.ax_rms.set_ylim(0, peak * 1.2)
        for rect, value, label, text in zip(self.bar_rects, values, quality_labels, self.quality_texts):
            rect.set_height(float(value))
            color = self._quality_color(label)
            rect.set_color(color)
            text.set_text(label)
            text.set_color(color)
            text.set_y(float(value) + 5)

    def update_status(self, status: str, recording: str, inference: str, prediction: str) -> None:
        self.status_text.set_text(status)
        self.recording_text.set_text(recording)
        self.inference_text.set_text(inference)
        self.prediction_text.set_text(prediction)

    def save_screenshot(self, directory: str = "screenshots") -> Path:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%dT%H%M%S")
        path = out_dir / f"screenshot_{timestamp}.png"
        self.fig.savefig(path, dpi=150)
        return path

    @staticmethod
    def _quality_color(label: str) -> str:
        mapping = {
            "GOOD": "#2ca02c",
            "WEAK": "#ff7f0e",
            "NOISY": "#d62728",
        }
        return mapping.get(label.upper(), "#1f77b4")
