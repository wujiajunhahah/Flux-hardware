"""Serial and synthetic EMG data sources."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

try:  # pragma: no cover - optional at import time
    import serial  # type: ignore
except Exception:  # pragma: no cover
    serial = None


HEADER = b"\xD2\xD2\xD2"
PACKET_BYTES = 29
EMG_FLAG = 0xAA
IMU_FLAG = 0xBB
CHANNEL_COUNT = 8


@dataclass
class EMGSample:
    """Represents one timestamped multi-channel EMG reading."""

    timestamp: float
    values: np.ndarray


@dataclass
class FrameStats:
    total_frames: int = 0
    emg_frames: int = 0
    imu_frames: int = 0
    dropped_frames: int = 0
    last_sequence: Optional[int] = None


def _decode_signed24(b1: int, b2: int, b3: int) -> int:
    value = (b1 << 16) | (b2 << 8) | b3
    if value & 0x800000:
        value -= 1 << 24
    return value


class BaseEMGStream:
    """Interface for EMG data providers."""

    def start(self) -> None:  # pragma: no cover - runtime wiring
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - runtime wiring
        raise NotImplementedError

    def consume_samples(self, max_items: int = 256) -> List[EMGSample]:  # pragma: no cover - runtime wiring
        raise NotImplementedError

    def frame_stats(self) -> FrameStats:  # pragma: no cover - runtime wiring
        raise NotImplementedError

    def send_command(self, data: str) -> None:  # pragma: no cover - runtime wiring
        """Optional hook to send a command back to the device."""


class SerialEMGStream(BaseEMGStream):
    """Read Waveletech EMG packets from a serial interface."""

    def __init__(self, port: str, baudrate: int = 921_600) -> None:
        if serial is None:
            raise RuntimeError("pyserial is required for hardware streaming")

        self.port = port
        self.baudrate = baudrate
        self._serial: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._buffer = bytearray()
        self._queue: Deque[EMGSample] = deque(maxlen=8096)
        self._lock = threading.Lock()
        self._stats = FrameStats()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        if self._serial and self._serial.is_open:
            self._serial.close()

    def _run(self) -> None:
        assert serial is not None
        try:
            conn = serial.Serial(self.port, self.baudrate, timeout=0.01)
        except Exception as exc:  # pragma: no cover - hardware side effect
            print(f"[stream] Failed to open {self.port}: {exc}")
            self._running = False
            return

        self._serial = conn

        while self._running:
            try:
                data = conn.read(conn.in_waiting or PACKET_BYTES)
            except Exception as exc:  # pragma: no cover
                print(f"[stream] Serial read error: {exc}")
                break
            if data:
                self._buffer.extend(data)
                self._process_buffer()
            else:
                time.sleep(0.001)

    def _process_buffer(self) -> None:
        while True:
            idx = self._buffer.find(HEADER)
            if idx == -1:
                if len(self._buffer) > len(HEADER):
                    del self._buffer[:- (len(HEADER) - 1)]
                return
            if len(self._buffer) - idx < PACKET_BYTES:
                if idx:
                    del self._buffer[:idx]
                return

            frame = self._buffer[idx : idx + PACKET_BYTES]
            del self._buffer[: idx + PACKET_BYTES]
            self._handle_frame(frame)

    def _handle_frame(self, frame: bytes) -> None:
        stats = self._stats
        stats.total_frames += 1
        frame_type = frame[3]
        sequence = frame[4]

        if stats.last_sequence is not None and frame_type == EMG_FLAG:
            expected = (stats.last_sequence + 1) & 0xFF
            if sequence != expected:
                delta = (sequence - expected) & 0xFF
                if delta <= 0:
                    delta = 1
                stats.dropped_frames += delta
        if frame_type == EMG_FLAG:
            stats.last_sequence = sequence

        if frame_type == EMG_FLAG:
            stats.emg_frames += 1
            payload = frame[5 : 5 + CHANNEL_COUNT * 3]
            values = np.zeros(CHANNEL_COUNT, dtype=float)
            for idx in range(CHANNEL_COUNT):
                b1, b2, b3 = payload[idx * 3 : idx * 3 + 3]
                values[idx] = _decode_signed24(b1, b2, b3)

            with self._lock:
                self._queue.append(EMGSample(time.perf_counter(), values))
        elif frame_type == IMU_FLAG:
            stats.imu_frames += 1
        else:
            # Unknown packet, drop silently.
            return

    def consume_samples(self, max_items: int = 256) -> List[EMGSample]:
        with self._lock:
            if max_items <= 0:
                items = list(self._queue)
                self._queue.clear()
                return items

            items: List[EMGSample] = []
            for _ in range(min(max_items, len(self._queue))):
                items.append(self._queue.popleft())
            return items

    def frame_stats(self) -> FrameStats:
        return self._stats

    def send_command(self, data: str) -> None:
        if not data:
            return
        conn = self._serial
        if conn is None or not conn.is_open:
            return
        try:
            payload = data.encode("utf-8")
            conn.write(payload)
        except Exception as exc:  # pragma: no cover
            print(f"[stream] Failed to send command: {exc}")


class SyntheticEMGStream(BaseEMGStream):
    """Generate pseudo EMG data for demo/testing."""

    def __init__(self, sample_rate: int = 1_000) -> None:
        self.sample_rate = sample_rate
        self._queue: Deque[EMGSample] = deque(maxlen=8096)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = FrameStats()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)

    def _run(self) -> None:
        rng = np.random.default_rng()
        t = 0.0
        dt = 1.0 / self.sample_rate
        while self._running:
            values = np.zeros(CHANNEL_COUNT)
            for ch in range(CHANNEL_COUNT):
                baseline = 50.0 * math.sin(2 * math.pi * (0.2 + ch * 0.05) * t)
                burst = 300.0 * math.sin(2 * math.pi * (1.2 + ch * 0.1) * t)
                envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 0.1 * t + ch * 0.3)
                noise = rng.normal(0, 20)
                values[ch] = baseline + envelope * burst + noise

            sample = EMGSample(time.perf_counter(), values)
            with self._lock:
                self._queue.append(sample)

            t += dt
            time.sleep(dt)

    def consume_samples(self, max_items: int = 256) -> List[EMGSample]:
        with self._lock:
            if max_items <= 0:
                items = list(self._queue)
                self._queue.clear()
                return items

            items: List[EMGSample] = []
            for _ in range(min(max_items, len(self._queue))):
                items.append(self._queue.popleft())
            return items

    def frame_stats(self) -> FrameStats:
        return self._stats

    def send_command(self, data: str) -> None:
        # Synthetic stream has nothing to send to.
        return
