"""UDP bridge to feed SerialEMGStream data into LibEMG."""

from __future__ import annotations

import json
import socket
import threading
from queue import Empty, Queue
from typing import Iterable

from .stream import EMGSample


class LibEMGBridge:
    """Asynchronously ships EMG samples to LibEMG's UDP handler."""

    def __init__(self, host: str = "127.0.0.1", port: int = 12345) -> None:
        self.host = host
        self.port = port
        self._queue: "Queue[EMGSample]" = Queue(maxsize=8192)
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, samples: Iterable[EMGSample]) -> None:
        for sample in samples:
            try:
                self._queue.put_nowait(sample)
            except Exception:
                # Drop samples if the queue is saturated; LibEMG can handle gaps.
                break

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self._queue.get(timeout=0.1)
            except Empty:
                continue

            payload = {
                "timestamp": sample.timestamp,
                "values": sample.values.tolist(),
            }
            message = json.dumps(payload).encode("utf-8")
            try:
                self._sock.sendto(message, (self.host, self.port))
            except OSError:
                # Ignore transient socket errors to keep bridge alive.
                continue


def stream_to_libemg(host: str = "127.0.0.1", port: int = 12345) -> LibEMGBridge:
    """Convenience helper returning a running LibEMGBridge."""

    return LibEMGBridge(host=host, port=port)
