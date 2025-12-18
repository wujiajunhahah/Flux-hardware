#!/usr/bin/env python3
"""Real-time 8-channel EMG monitor with waveform + RMS bars.

This script decodes Waveletech EMG packets streamed over a serial port
and displays both the raw waveforms and 200 ms RMS indicators. If a
serial port is not provided, a clean synthetic generator is used so the
visualization can run as a demo.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from collections import deque
from typing import Deque, Iterable, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    import serial  # type: ignore
except ImportError:  # pragma: no cover - serial is optional for demo mode
    serial = None


HEADER = b"\xD2\xD2\xD2"
PACKET_BYTES = 29
EMG_FLAG = 0xAA
CHANNEL_COUNT = 8


def _apply_plot_style() -> None:
    """Apply a Matplotlib style that works across versions."""

    preferred_styles = [
        "seaborn-v0_8-whitegrid",
        "seaborn-whitegrid",
        "seaborn-v0_8",
        "seaborn",
        "ggplot",
    ]
    for style in preferred_styles:
        if style in plt.style.available:
            plt.style.use(style)
            return

    plt.style.use("default")


def _decode_signed24(raw: bytes) -> float:
    """Decode a signed 24-bit little-endian value to millivolts."""

    if len(raw) != 3:
        raise ValueError("signed24 expects exactly 3 bytes")

    value = raw[0] | (raw[1] << 8) | (raw[2] << 16)
    if value & 0x800000:
        value -= 1 << 24

    # The device reports microvolts; convert to millivolts for readability.
    return value * 1e-3


class WaveletechEMGSource:
    """Stream EMG samples from a Waveletech wearable over serial."""

    def __init__(self, port: str, baudrate: int = 921_600):
        if serial is None:
            raise RuntimeError("pyserial is required for hardware streaming")

        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        self.is_running = False
        self.lock = threading.Lock()
        self.buffer: Deque[np.ndarray] = deque(maxlen=2048)
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.serial and self.serial.is_open:
            self.serial.close()

    def _run(self) -> None:
        try:
            conn = serial.Serial(self.port, self.baudrate, timeout=0.01)
        except Exception as exc:  # pragma: no cover - hardware-specific error
            print(f"Failed to open {self.port}: {exc}")
            self.is_running = False
            return

        with conn:
            self.serial = conn
            frame = bytearray()

            while self.is_running:
                data = conn.read(conn.in_waiting or PACKET_BYTES)
                if not data:
                    continue
                frame.extend(data)

                while True:
                    idx = frame.find(HEADER)
                    if idx == -1:
                        if len(frame) > len(HEADER):
                            del frame[:- (len(HEADER) - 1)]
                        break
                    if len(frame) - idx < PACKET_BYTES:
                        # Wait for more bytes
                        if idx:
                            del frame[:idx]
                        break

                    packet = frame[idx:idx + PACKET_BYTES]
                    del frame[:idx + PACKET_BYTES]

                    if packet[3] != EMG_FLAG:
                        continue  # skip non-EMG frames (BB = IMU)

                    payload = packet[5:5 + CHANNEL_COUNT * 3]
                    sample = []
                    for ch in range(CHANNEL_COUNT):
                        raw = payload[ch * 3: (ch + 1) * 3]
                        sample.append(_decode_signed24(raw))

                    with self.lock:
                        self.buffer.append(np.array(sample, dtype=float))

    def get_samples(self, max_items: int = 32) -> List[np.ndarray]:
        """Retrieve up to ``max_items`` recent samples."""

        with self.lock:
            items = list(self.buffer)
            self.buffer.clear()
        if max_items:
            return items[-max_items:]
        return items


class SyntheticEMGSource:
    """Fallback generator for demoing visualization without hardware."""

    def __init__(self, sample_rate: int = 1_000, channels: int = CHANNEL_COUNT):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_running = False
        self.lock = threading.Lock()
        self.buffer: Deque[np.ndarray] = deque(maxlen=2048)
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1)

    def _run(self) -> None:
        t = 0.0
        dt = 1.0 / self.sample_rate
        rng = np.random.default_rng()

        while self.is_running:
            sample = np.zeros(self.channels)
            for ch in range(self.channels):
                noise = rng.normal(0.0, 5.0)
                burst = math.sin(2 * math.pi * 0.8 * t + ch * 0.2)
                envelope = 1.0 if rng.random() > 0.99 else 0.1
                power_line = 20.0 * math.sin(2 * math.pi * 50 * t)
                sample[ch] = noise + envelope * burst * 50 + power_line

            with self.lock:
                self.buffer.append(sample)

            t += dt
            time.sleep(dt)

    def get_samples(self, max_items: int = 32) -> List[np.ndarray]:
        with self.lock:
            items = list(self.buffer)
            self.buffer.clear()
        return items[-max_items:]


class EMGVisualizer:
    """Plot multi-channel EMG waveforms + 200 ms RMS indicators."""

    def __init__(
        self,
        source,
        sample_rate: int = 1_000,
        window_seconds: float = 3.0,
        rms_window: float = 0.2,
    ) -> None:
        self.source = source
        self.sample_rate = sample_rate
        self.window_samples = int(window_seconds * sample_rate)
        self.rms_samples = max(1, int(rms_window * sample_rate))
        self.time_buffer: Deque[float] = deque(maxlen=self.window_samples)
        self.emg_buffers: List[Deque[float]] = [
            deque(maxlen=self.window_samples) for _ in range(CHANNEL_COUNT)
        ]
        self.start_time = time.perf_counter()

        # Prepare figure
        _apply_plot_style()
        self.fig = plt.figure(figsize=(12, 7))
        grid = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])
        self.ax_wave = self.fig.add_subplot(grid[0, 0])
        self.ax_rms = self.fig.add_subplot(grid[1, 0])

        self.offset = np.arange(CHANNEL_COUNT) * 250.0
        self.lines = []
        for idx in range(CHANNEL_COUNT):
            (line,) = self.ax_wave.plot([], [], lw=1.2, label=f"CH{idx + 1}")
            self.lines.append(line)

        self.ax_wave.set_yticks(self.offset)
        self.ax_wave.set_yticklabels([f"CH{idx+1}" for idx in range(CHANNEL_COUNT)])
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_ylabel("Amplitude (mV)")
        self.ax_wave.set_title("Real-time EMG (offset stacked)")
        self.ax_wave.set_ylim(-200, self.offset[-1] + 200)

        x_positions = np.arange(1, CHANNEL_COUNT + 1)
        self.bars = self.ax_rms.bar(x_positions, np.zeros(CHANNEL_COUNT))
        self.ax_rms.set_xticks(x_positions)
        self.ax_rms.set_xticklabels([f"CH{idx+1}" for idx in range(CHANNEL_COUNT)])
        self.ax_rms.set_ylabel("RMS (mV)")
        self.ax_rms.set_title("200 ms RMS")
        self.text_labels = [self.ax_rms.text(x, 0, "0.0", ha="center", va="bottom") for x in x_positions]
        self.ax_rms.set_ylim(0, 200)

    def _append_samples(self, samples: Iterable[np.ndarray]) -> None:
        for sample in samples:
            ts = time.perf_counter() - self.start_time
            self.time_buffer.append(ts)
            for ch in range(CHANNEL_COUNT):
                self.emg_buffers[ch].append(sample[ch])

    def _update_waveforms(self) -> None:
        if not self.time_buffer:
            return
        times = np.array(self.time_buffer)
        window = self.window_samples / self.sample_rate
        t0 = max(0.0, times[-1] - window)
        times = times - t0

        for idx, line in enumerate(self.lines):
            samples = np.array(self.emg_buffers[idx])
            line.set_data(times[-len(samples):], samples + self.offset[idx])

        self.ax_wave.set_xlim(0, max(window, times[-1]))

    def _update_rms(self) -> None:
        max_rms = self.ax_rms.get_ylim()[1]
        resized = False

        for idx in range(CHANNEL_COUNT):
            buffer = np.array(self.emg_buffers[idx])
            if buffer.size < self.rms_samples:
                rms = 0.0
            else:
                rms = float(np.sqrt(np.mean(buffer[-self.rms_samples :] ** 2)))

            self.bars[idx].set_height(rms)
            self.text_labels[idx].set_position((idx + 1, rms))
            self.text_labels[idx].set_text(f"{rms:0.1f}")

            if rms > max_rms * 0.95:
                max_rms = rms * 1.4
                resized = True

        if resized:
            self.ax_rms.set_ylim(0, max_rms)

    def _tick(self, _frame):
        samples = self.source.get_samples()
        if samples:
            self._append_samples(samples)
            self._update_waveforms()
            self._update_rms()
        return (*self.lines, *self.bars, *self.text_labels)

    def run(self) -> None:
        self.source.start()
        self.anim = FuncAnimation(self.fig, self._tick, interval=40, blit=True)
        self.fig.tight_layout()
        plt.show()
        if hasattr(self.source, "stop"):
            self.source.stop()


def build_source(port: Optional[str], baudrate: int) -> object:
    if port:
        return WaveletechEMGSource(port, baudrate)
    return SyntheticEMGSource()


def main() -> None:
    parser = argparse.ArgumentParser(description="8-channel EMG real-time RMS visualizer")
    parser.add_argument("--port", help="Serial port connected to the wearable", default=None)
    parser.add_argument("--baudrate", type=int, default=921600, help="Serial baudrate (default: 921600)")
    parser.add_argument("--window", type=float, default=3.0, help="Waveform window length in seconds")
    parser.add_argument("--rms", type=float, default=0.2, help="RMS window length in seconds")
    args = parser.parse_args()

    source = build_source(args.port, args.baudrate)
    visualizer = EMGVisualizer(source, sample_rate=1_000, window_seconds=args.window, rms_window=args.rms)
    visualizer.run()


if __name__ == "__main__":
    main()
