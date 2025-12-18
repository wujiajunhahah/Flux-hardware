"""Matplotlib-based visualization for the EMG gesture tool."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox

from .stream import CHANNEL_COUNT


class EMGPlotUI:
    """Handles waveform, RMS, and status panel rendering."""

    def __init__(self, window_seconds: float, interval_ms: int = 50) -> None:
        plt.style.use("default")
        self.window_seconds = window_seconds
        self.interval_ms = interval_ms

        self.fig = plt.figure(figsize=(12, 8))
        grid = self.fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 0.5], hspace=0.35)

        self.ax_wave = self.fig.add_subplot(grid[0])
        self.ax_rms = self.fig.add_subplot(grid[1])
        self.ax_status = self.fig.add_subplot(grid[2])
        self.ax_status.axis("off")

        # Waveform setup
        self.lines = [self.ax_wave.plot([], [], lw=1)[0] for _ in range(CHANNEL_COUNT)]
        self.ax_wave.set_ylabel("EMG (normalized)")
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_title("Waveletech EMG - Project Flux")

        # RMS bar setup
        x = np.arange(CHANNEL_COUNT)
        self.bar_rects = self.ax_rms.bar(x, np.zeros(CHANNEL_COUNT), color="royalblue")
        self.ax_rms.set_xticks(x)
        self.ax_rms.set_xticklabels([f"Ch {i+1}" for i in range(CHANNEL_COUNT)])
        self.ax_rms.set_ylabel("RMS (µV)")
        self.ax_rms.set_ylim(0, 300)

        self.quality_texts = [
            self.ax_rms.text(idx, 0, "", ha="center", va="bottom", fontsize=9) for idx in x
        ]

        # Status texts
        self.status_text = self.ax_status.text(
            0.01, 0.85, "", transform=self.ax_status.transAxes, fontsize=10
        )
        self.recording_text = self.ax_status.text(
            0.01, 0.6, "", transform=self.ax_status.transAxes, fontsize=10
        )
        self.inference_text = self.ax_status.text(
            0.01, 0.35, "", transform=self.ax_status.transAxes, fontsize=10
        )
        self.prediction_text = self.ax_wave.text(
            0.99,
            0.95,
            "",
            transform=self.ax_wave.transAxes,
            ha="right",
            fontsize=13,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        self.flux_text = self.ax_status.text(
            0.01, 0.1, "", transform=self.ax_status.transAxes, fontsize=10, color="#1f77b4"
        )

        self.anim: FuncAnimation | None = None
        self._actions = {
            "record": None,
            "train": None,
            "inference": None,
            "screenshot": None,
            "quit": None,
        }
        self._label_box: TextBox | None = None
        self._button_map: dict[str, Button] = {}
        self._init_buttons()

    def connect_keypress(self, callback) -> None:
        self.fig.canvas.mpl_connect("key_press_event", callback)

    def set_actions(
        self,
        record=None,
        train=None,
        inference=None,
        screenshot=None,
        quit=None,
    ) -> None:
        self._actions.update(
            {
                "record": record,
                "train": train,
                "inference": inference,
                "screenshot": screenshot,
                "quit": quit,
            }
        )

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

        offsets = np.arange(CHANNEL_COUNT) * 1.5
        shifted = data + offsets[:, None]

        for idx, line in enumerate(self.lines):
            line.set_data(times, shifted[idx])

        self.ax_wave.set_ylim(-0.5, offsets[-1] + 2.0)
        self.ax_wave.set_yticks(offsets)
        self.ax_wave.set_yticklabels([f"Ch {idx+1}" for idx in range(CHANNEL_COUNT)])

    def update_rms(self, rms_values: Sequence[float], quality_labels: Sequence[str]) -> None:
        # --- FIX START: Safely handle numpy arrays ---
        values = np.asarray(rms_values, dtype=float)
        if values.size == 0:
            values = np.zeros(len(self.bar_rects))
        
        peak = max(50.0, float(np.max(values)))
        # --- FIX END ---
        
        self.ax_rms.set_ylim(0, peak * 1.2)

        for rect, value, label, text in zip(
            self.bar_rects, values, quality_labels, self.quality_texts
        ):
            rect.set_height(float(value))
            color = self._quality_color(label)
            rect.set_color(color)
            text.set_text(label)
            text.set_color(color)
            text.set_y(float(value) + 5)

    def update_status(
        self,
        status: str,
        recording: str,
        inference: str,
        prediction: str,
        flux_text: str,
    ) -> None:
        self.status_text.set_text(status)
        self.recording_text.set_text(recording)
        self.inference_text.set_text(inference)
        self.prediction_text.set_text(prediction)
        self.flux_text.set_text(flux_text)

    def save_screenshot(self, directory: str = "screenshots") -> Path:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%dT%H%M%S")
        path = out_dir / f"screenshot_{timestamp}.png"
        self.fig.savefig(path, dpi=150)
        return path

    def get_label_text(self) -> str:
        if not self._label_box:
            return ""
        return self._label_box.text.strip()

    def set_record_button_state(self, is_recording: bool, label: str) -> None:
        button = self._button_map.get("record")
        if not button:
            return
        new_label = "Stop" if is_recording else "Record"
        if is_recording and label:
            new_label = f"Stop ({label})"
        button.label.set_text(new_label)

    @staticmethod
    def _quality_color(label: str) -> str:
        mapping = {
            "GOOD": "#2ca02c",
            "WEAK": "#ff7f0e",
            "NOISY": "#d62728",
        }
        return mapping.get(label.upper(), "#1f77b4")

    def _init_buttons(self) -> None:
        label_ax = self.fig.add_axes([0.05, 0.08, 0.15, 0.05])
        self._label_box = TextBox(label_ax, "Label", initial="rest")

        button_specs = [
            ("record", "Record", self._handle_record, 0.23),
            ("train", "Train", self._handle_train, 0.36),
            ("inference", "Inference", self._handle_inference, 0.49),
            ("screenshot", "Screenshot", self._handle_screenshot, 0.62),
            ("quit", "Quit", self._handle_quit, 0.75),
        ]
        for key, label, handler, x in button_specs:
            ax = self.fig.add_axes([x, 0.02, 0.1, 0.05])
            button = Button(ax, label)
            button.on_clicked(handler)
            self._button_map[key] = button

    def _call_action(self, name: str, *args, **kwargs) -> None:
        action = self._actions.get(name)
        if action:
            action(*args, **kwargs)

    def _handle_record(self, event) -> None:  # noqa: D401
        self._call_action("record", label=self.get_label_text())

    def _handle_train(self, event) -> None:
        self._call_action("train")

    def _handle_inference(self, event) -> None:
        self._call_action("inference")

    def _handle_screenshot(self, event) -> None:
        self._call_action("screenshot")

    def _handle_quit(self, event) -> None:
        self._call_action("quit")
