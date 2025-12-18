"""Serial and synthetic EMG data sources."""

from __future__ import annotations

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
class IMUSample:
    """Represents one timestamped IMU reading (accel + gyro)."""

    timestamp: float
    accel: np.ndarray
    gyro: np.ndarray


@dataclass
class FrameStats:
    total_frames: int = 0
    emg_frames: int = 0
    imu_frames: int = 0
    dropped_frames: int = 0
    last_sequence: Optional[int] = None


def _decode_signed24(b1: int, b2: int, b3: int) -> float:
    value = (b1 << 16) | (b2 << 8) | b3
    if value & 0x800000:
        value -= 1 << 24

    # Convert to volts then microvolts
    # V_ref = 4.5V, Gain = 1200
    volts = (value / 8388607.0) * 4.5
    micro_volts = (volts / 1200.0) * 1_000_000.0
    return float(micro_volts)


class BaseEMGStream:
    """Interface for EMG data providers."""

    def start(self) -> None:  # pragma: no cover - runtime wiring
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - runtime wiring
        raise NotImplementedError

    def consume_samples(self, max_items: int = 256) -> List[EMGSample]:  # pragma: no cover - runtime wiring
        raise NotImplementedError

    def consume_imu(self, max_items: int = 256) -> List[IMUSample]:  # pragma: no cover - runtime wiring
        return []

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
        self._imu_queue: Deque[IMUSample] = deque(maxlen=4096)
        self._lock = threading.Lock()
        self._imu_lock = threading.Lock()
        self._stats = FrameStats()

    def start(self) -> None:
        if self._running:
            return
        self._verify_port()
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
            payload = frame[5 : 5 + 12]
            accel = np.zeros(3, dtype=float)
            gyro = np.zeros(3, dtype=float)
            for idx in range(3):
                start = idx * 2
                accel[idx] = int.from_bytes(payload[start : start + 2], "little", signed=True)
            for idx in range(3):
                start = 3 * 2 + idx * 2
                gyro[idx] = int.from_bytes(payload[start : start + 2], "little", signed=True)

            sample = IMUSample(time.perf_counter(), accel, gyro)
            with self._imu_lock:
                self._imu_queue.append(sample)
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

    def consume_imu(self, max_items: int = 256) -> List[IMUSample]:
        with self._imu_lock:
            if max_items <= 0:
                items = list(self._imu_queue)
                self._imu_queue.clear()
                return items

            items: List[IMUSample] = []
            for _ in range(min(max_items, len(self._imu_queue))):
                items.append(self._imu_queue.popleft())
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

    def _verify_port(self) -> None:
        if serial is None:
            raise RuntimeError("pyserial is required for hardware streaming")
        try:
            conn = serial.Serial(self.port, self.baudrate, timeout=0.01)
        except Exception as exc:
            raise RuntimeError(f"Unable to open {self.port}: {exc}") from exc
        else:
            conn.close()
