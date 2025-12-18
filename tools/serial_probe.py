#!/usr/bin/env python3
"""Quick sanity probe for Waveletech EMG serial data."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import serial

# --- FIX START: Auto-detect project root ---
# Resolve the project root relative to this script (tools/../)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- FIX END ---

from src.stream import EMG_FLAG, HEADER, IMU_FLAG, PACKET_BYTES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Waveletech serial probe")
    parser.add_argument("--port", required=True, help="Serial port (e.g., /dev/cu.usbserial-0001)")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate")
    parser.add_argument("--seconds", type=int, default=5, help="Seconds to sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = {
        "total": 0,
        "emg": 0,
        "imu": 0,
        "other": 0,
        "dropped": 0,
        "last_seq": None,
    }
    buffer = bytearray()
    print(f"[probe] Listening on {args.port} @ {args.baud} baud for {args.seconds}s ...")
    start = time.time()
    try:
        with serial.Serial(args.port, args.baud, timeout=0.01) as conn:
            while time.time() - start < args.seconds:
                data = conn.read(conn.in_waiting or PACKET_BYTES)
                if not data:
                    continue
                buffer.extend(data)
                _process_buffer(buffer, stats)
    except KeyboardInterrupt:
        print("\n[probe] Interrupted by user")
    except Exception as e:
        print(f"\n[probe] Error: {e}")

    duration = max(0.001, time.time() - start)
    emg_ratio = stats["emg"] / stats["total"] if stats["total"] else 0.0
    imu_ratio = stats["imu"] / stats["total"] if stats["total"] else 0.0
    print("[probe] Summary")
    print(
        f"  Frames: {stats['total']} total | {stats['emg']} EMG (AA) | "
        f"{stats['imu']} IMU (BB) | {stats['other']} other"
    )
    print(f"  Ratios: {emg_ratio:.2%} EMG | {imu_ratio:.2%} IMU")
    print(f"  Dropped SN: {stats['dropped']}")
    print(f"  Throughput: {stats['emg']/duration:.1f} EMG frames/sec")


def _process_buffer(buffer: bytearray, stats: dict) -> None:
    while True:
        idx = buffer.find(HEADER)
        if idx == -1:
            if len(buffer) > len(HEADER):
                del buffer[:- (len(HEADER) - 1)]
            return
        if len(buffer) - idx < PACKET_BYTES:
            return

        frame = buffer[idx : idx + PACKET_BYTES]
        del buffer[: idx + PACKET_BYTES]
        stats["total"] += 1
        frame_type = frame[3]
        seq = frame[4]

        if frame_type == EMG_FLAG:
            stats["emg"] += 1
            _update_sequence(stats, seq)
        elif frame_type == IMU_FLAG:
            stats["imu"] += 1
        else:
            stats["other"] += 1


def _update_sequence(stats: dict, seq: int) -> None:
    last = stats.get("last_seq")
    if last is not None:
        expected = (last + 1) & 0xFF
        if seq != expected:
            delta = (seq - expected) & 0xFF
            if delta <= 0:
                delta = 1
            stats["dropped"] += delta
    stats["last_seq"] = seq


if __name__ == "__main__":
    main()
