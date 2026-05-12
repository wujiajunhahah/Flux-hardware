"""Serial, BLE, and synthetic EMG data sources."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import sys
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

try:  # pragma: no cover
    from bleak import BleakClient, BleakScanner  # type: ignore
except Exception:  # pragma: no cover
    BleakClient = None  # type: ignore
    BleakScanner = None  # type: ignore


HEADER = b"\xD2\xD2\xD2"
PACKET_BYTES = 29
EMG_FLAG = 0xAA
IMU_FLAG = 0xBB
CHANNEL_COUNT = 8

BLE_SERVICE_UUID = "974cbe30-3e83-465e-acde-6f92fe712134"
BLE_DATA_CHAR = "974cbe31-3e83-465e-acde-6f92fe712134"
BLE_WRITE_CHAR = "974cbe32-3e83-465e-acde-6f92fe712134"
BLE_DEVICE_PREFIX = "WL"


def _is_ignorable_modem_control_error(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and getattr(exc, "errno", None) in {
        errno.EPERM,
        errno.EINVAL,
        errno.ENOTTY,
    }


if serial is not None:
    class CompatSerial(serial.Serial):
        """Tolerate USB serial bridges that reject modem-control ioctls."""

        def _update_dtr_state(self):  # type: ignore[override]
            try:
                super()._update_dtr_state()
            except OSError as exc:
                if not _is_ignorable_modem_control_error(exc):
                    raise

        def _update_rts_state(self):  # type: ignore[override]
            try:
                super()._update_rts_state()
            except OSError as exc:
                if not _is_ignorable_modem_control_error(exc):
                    raise
else:  # pragma: no cover - exercised only when pyserial is unavailable
    CompatSerial = None


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
    dropped_frames: int = 0        # 仅 EMG 帧丢包
    last_sequence: Optional[int] = None        # 全局 sequence (向后兼容)
    last_emg_sequence: Optional[int] = None    # EMG 独立 sequence 追踪
    last_imu_sequence: Optional[int] = None    # IMU 独立 sequence 追踪


def _decode_signed24(b1: int, b2: int, b3: int) -> float:
    """Decode 3-byte big-endian signed integer.

    Per WAVELETECH manual: the 24-bit value is already in microvolts (µV).
    No V_ref/Gain conversion needed — the firmware handles ADC scaling internally.
    """
    value = (b1 << 16) | (b2 << 8) | b3
    if value & 0x800000:
        value -= 1 << 24
    return float(value)


def normalize_serial_device(path: str) -> str:
    """
    macOS：读线程用 /dev/tty.* 常出现能打开但收不到流或行为异常，
    与对应 /dev/cu.* 等效时应优先 cu.*。
    """
    p = (path or "").strip()
    if not p:
        return p
    if sys.platform == "darwin" and p.startswith("/dev/tty."):
        return "/dev/cu." + p[len("/dev/tty.") :]
    return p


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

        self.port = normalize_serial_device(port)
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
            assert CompatSerial is not None
            conn = CompatSerial(self.port, self.baudrate, timeout=0.01)
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

        # 丢包检测: 设备使用全局 sequence counter (EMG+IMU 共享)
        # 只要全局序号不连续即为真丢包
        if stats.last_sequence is not None:
            expected = (stats.last_sequence + 1) & 0xFF
            if sequence != expected:
                delta = (sequence - expected) & 0xFF
                if delta <= 0:
                    delta = 1
                stats.dropped_frames += delta
        stats.last_sequence = sequence

        # 按类型追踪 (诊断用)
        if frame_type == EMG_FLAG:
            stats.last_emg_sequence = sequence
        elif frame_type == IMU_FLAG:
            stats.last_imu_sequence = sequence

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
            # Per WAVELETECH manual: IMU is big-endian, layout is gyro(3x2) then accel(3x2)
            # Scale factors: gyro × 0.0012 → rad/s, accel × 0.0005978 → m/s²
            payload = frame[5 : 5 + 12]
            accel = np.zeros(3, dtype=float)
            gyro = np.zeros(3, dtype=float)
            for idx in range(3):
                start = idx * 2
                raw = int.from_bytes(payload[start : start + 2], "big", signed=True)
                gyro[idx] = raw * 0.0012  # rad/s
            for idx in range(3):
                start = 6 + idx * 2
                raw = int.from_bytes(payload[start : start + 2], "big", signed=True)
                accel[idx] = raw * 0.0005978  # m/s²

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
            assert CompatSerial is not None
            conn = CompatSerial(self.port, self.baudrate, timeout=0.01)
        except Exception as exc:
            raise RuntimeError(f"Unable to open {self.port}: {exc}") from exc
        else:
            conn.close()


# ─── BLE Stream ──────────────────────────────────────────────


class BleEMGStream(BaseEMGStream):
    """Read Waveletech EMG packets directly via BLE (no USB dongle)."""

    def __init__(self, address: Optional[str] = None) -> None:
        if BleakClient is None:
            raise RuntimeError("bleak is required: pip install bleak")
        # 防御：构造时如果有人传字面量 "auto"，立即归一化成 None 触发扫描
        if address == "auto":
            import traceback
            print(f"[ble] WARN: BleEMGStream constructed with address='auto', falling back to scan")
            print("[ble] Stack trace:")
            traceback.print_stack()
            address = None
        self._address = address
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._queue: Deque[EMGSample] = deque(maxlen=8096)
        self._imu_queue: Deque[IMUSample] = deque(maxlen=4096)
        self._lock = threading.Lock()
        self._imu_lock = threading.Lock()
        self._stats = FrameStats()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = False
        # debug：让 instance id 出现在日志里，方便看到底有几个 BleEMGStream 实例
        import os
        self._inst_id = f"{os.getpid()}-{id(self) & 0xFFFFFF:x}"

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ble_main())
        except Exception as exc:
            print(f"[ble] Event loop fatal: {exc}")
        finally:
            self._running = False

    async def _ble_main(self) -> None:
        """主循环：扫描一次 + 多次重连。每次连接断开后退避重试。

        修复点：
        - 扫描成功后 cache 到 self._address，避免空地址重连去拼接成 "auto"
        - 用 disconnect callback + asyncio.Event 监听断开，比 mtu_size 轮询稳
        - 每次连接独立 try/except 隔离单次异常，不让单次失败干掉整个线程
        - 退避：1s → 2s → 4s → 8s 上限，避免狂打蓝牙栈
        """
        if not self._address:
            scanned = await self._scan_for_wristband()
            if not scanned:
                print("[ble] Wristband not found. Make sure it's ON and USB dongle is UNPLUGGED.")
                return
            self._address = scanned  # 缓存，后续重连复用

        backoff = 1.0
        max_backoff = 8.0
        attempt = 0

        while self._running:
            attempt += 1
            print(f"[ble inst={self._inst_id}] Connecting to {self._address} (attempt {attempt})...")
            disconnect_event = asyncio.Event()

            def _on_disconnect(_client) -> None:
                # CoreBluetooth 在断连时调用；切换 event 让 keep-alive 跳出
                if not disconnect_event.is_set():
                    print("[ble] Peripheral disconnected")
                    disconnect_event.set()

            try:
                async with BleakClient(self._address, disconnected_callback=_on_disconnect) as client:
                    self._connected = True
                    # 立即尝试加大 MTU 到 247（macOS 一般会协商到 185）；失败不致命
                    try:
                        await client._backend._acquire_mtu()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    print(f"[ble] Connected! MTU={client.mtu_size}")

                    await client.start_notify(BLE_DATA_CHAR, self._on_notify)
                    print(f"[ble] Subscribed to EMG data stream")

                    # 等待断开 OR 停止信号；不再用 sleep(0.1) 轮询
                    while self._running and not disconnect_event.is_set():
                        try:
                            await asyncio.wait_for(disconnect_event.wait(), timeout=0.5)
                            break
                        except asyncio.TimeoutError:
                            continue

                    try:
                        await client.stop_notify(BLE_DATA_CHAR)
                    except Exception:
                        pass  # 已经断开时 stop_notify 会失败，吞掉

                self._connected = False
                if not self._running:
                    print("[ble] Stopped by user")
                    return
                # 否则进入重连退避
                print(f"[ble] Reconnecting in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

            except asyncio.CancelledError:
                print("[ble] Cancelled")
                self._connected = False
                return
            except Exception as exc:
                self._connected = False
                print(f"[ble] Connection error: {exc!r} — retrying in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def _scan_for_wristband(self, timeout: float = 10.0) -> Optional[str]:
        print(f"[ble] Scanning for WAVELETECH wristband ({timeout}s)...")
        devices = await BleakScanner.discover(timeout=timeout, return_adv=True)

        for addr, (dev, adv) in devices.items():
            name = dev.name or adv.local_name or ""
            if name.upper().startswith(BLE_DEVICE_PREFIX):
                print(f"[ble] Found: {name} ({addr}) RSSI={adv.rssi}")
                return addr

        for addr, (dev, adv) in devices.items():
            uuids = adv.service_uuids or []
            if BLE_SERVICE_UUID in uuids:
                name = dev.name or "(unknown)"
                print(f"[ble] Found by UUID: {name} ({addr})")
                return addr

        return None

    def _on_notify(self, sender: int, data: bytearray) -> None:
        if len(data) < 20:
            return

        frame_type = data[0]
        seq = data[1]
        stats = self._stats
        stats.total_frames += 1

        # 全局序号丢包检测 (EMG+IMU 共享计数器)
        if stats.last_sequence is not None:
            expected = (stats.last_sequence + 1) & 0xFF
            if seq != expected:
                delta = (seq - expected) & 0xFF
                stats.dropped_frames += max(delta, 1)
        stats.last_sequence = seq

        if frame_type == EMG_FLAG:
            stats.emg_frames += 1
            stats.last_emg_sequence = seq

            values = np.zeros(CHANNEL_COUNT, dtype=float)
            payload = data[2:]  # 18 bytes
            n_channels = min(len(payload) // 3, CHANNEL_COUNT)
            for i in range(n_channels):
                b1, b2, b3 = payload[i * 3 : i * 3 + 3]
                values[i] = _decode_signed24(b1, b2, b3)

            with self._lock:
                self._queue.append(EMGSample(time.perf_counter(), values))

        elif frame_type == IMU_FLAG:
            stats.imu_frames += 1
            stats.last_imu_sequence = seq
            # Per WAVELETECH manual: IMU big-endian, gyro(3x2) then accel(3x2)
            payload = data[2:]
            if len(payload) >= 12:
                accel = np.zeros(3, dtype=float)
                gyro = np.zeros(3, dtype=float)
                for i in range(3):
                    s = i * 2
                    raw = int.from_bytes(payload[s : s + 2], "big", signed=True)
                    gyro[i] = raw * 0.0012  # rad/s
                for i in range(3):
                    s = 6 + i * 2
                    raw = int.from_bytes(payload[s : s + 2], "big", signed=True)
                    accel[i] = raw * 0.0005978  # m/s²
                with self._imu_lock:
                    self._imu_queue.append(IMUSample(time.perf_counter(), accel, gyro))

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
