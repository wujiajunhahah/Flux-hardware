"""
FluxChi — Web 后端
=========================================================
FastAPI + WebSocket 实时推送 EMG 状态到前端仪表盘。

用法：
  python web/app.py --port /dev/tty.usbserial-XXXX
  python web/app.py --demo   # 无硬件演示模式

浏览器打开: http://localhost:8000
=========================================================
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.stream import CHANNEL_COUNT, BaseEMGStream, SerialEMGStream
from src.fatigue import FatigueEstimator, FatigueReading
from src.decision import DecisionEngine, DecisionOutput

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="FluxChi", version="1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── Global state ────────────────────────────────────────────
stream: Optional[BaseEMGStream] = None
onnx_session = None
onnx_classes: List[str] = []
onnx_config: Optional[dict] = None
fatigue_estimator: Optional[FatigueEstimator] = None
decision_engine: Optional[DecisionEngine] = None
active_connections: Set[WebSocket] = set()
demo_mode = False
timeline: List[Dict] = []
SAMPLE_RATE = 1000
WINDOW_SEC = 0.25


# ─── ONNX Inference wrapper ─────────────────────────────────
class ONNXClassifier:
    """Wraps an ONNX model for activity classification."""

    def __init__(self, onnx_path: str, config: dict):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(onnx_path))
        self.input_name = self.session.get_inputs()[0].name
        self.classes = config.get("classes", [])
        self.n_features = config.get("n_features", 84)

    def predict(self, features: np.ndarray) -> tuple:
        """Return (class_label, confidence, probabilities_dict)."""
        x = features.astype(np.float32).reshape(1, -1)
        if x.shape[1] < self.n_features:
            x = np.pad(x, ((0, 0), (0, self.n_features - x.shape[1])))
        elif x.shape[1] > self.n_features:
            x = x[:, :self.n_features]

        outputs = self.session.run(None, {self.input_name: x})
        label = str(outputs[0][0])
        proba_list = outputs[1] if len(outputs) > 1 else None

        probabilities = {}
        confidence = 1.0
        if proba_list is not None:
            proba_dict = proba_list[0]
            if isinstance(proba_dict, dict):
                probabilities = {str(k): float(v) for k, v in proba_dict.items()}
                confidence = max(probabilities.values()) if probabilities else 1.0
            elif isinstance(proba_dict, (list, np.ndarray)):
                arr = np.array(proba_dict).flatten()
                for i, cls in enumerate(self.classes):
                    if i < len(arr):
                        probabilities[cls] = float(arr[i])
                confidence = float(np.max(arr)) if len(arr) > 0 else 1.0

        return label, confidence, probabilities


# ─── Enhanced feature extraction (84-dim, matches train_pipeline) ──
def extract_window_features_84(window: np.ndarray, sample_rate: int) -> np.ndarray:
    """84-dim features: 7 per channel (MAV,RMS,WL,ZC,SSC,MNF,MDF) + 28 correlations."""
    feats = []
    n_ch = min(window.shape[1], CHANNEL_COUNT)
    threshold = 20.0

    for ch in range(n_ch):
        sig = window[:, ch]
        mav = np.mean(np.abs(sig))
        rms = np.sqrt(np.mean(sig ** 2))
        wl = np.sum(np.abs(np.diff(sig))) if len(sig) > 1 else 0.0

        diffs = np.diff(np.sign(sig))
        abs_diff = np.abs(np.diff(sig))
        zc = int(np.sum((diffs != 0) & (abs_diff > threshold)))

        if len(sig) >= 3:
            d1 = sig[1:-1] - sig[:-2]
            d2 = sig[2:] - sig[1:-1]
            ssc = int(np.sum((d1 * d2 < 0) & (np.abs(d1) > threshold) & (np.abs(d2) > threshold)))
        else:
            ssc = 0

        if len(sig) >= 4:
            freqs = np.fft.rfftfreq(len(sig), d=1.0 / sample_rate)
            psd = np.abs(np.fft.rfft(sig)) ** 2
            total_power = np.sum(psd)
            if total_power > 1e-12:
                mnf = float(np.sum(freqs * psd) / total_power)
                cumulative = np.cumsum(psd)
                idx = np.searchsorted(cumulative, total_power / 2.0)
                mdf = float(freqs[min(idx, len(freqs) - 1)])
            else:
                mnf, mdf = 0.0, 0.0
        else:
            mnf, mdf = 0.0, 0.0

        feats.extend([mav, rms, wl, zc, ssc, mnf, mdf])

    if window.shape[0] >= 2:
        try:
            corr = np.corrcoef(window[:, :n_ch], rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
        except Exception:
            corr = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                feats.append(corr[i, j])
    else:
        n_pairs = n_ch * (n_ch - 1) // 2
        feats.extend([0.0] * n_pairs)

    return np.array(feats, dtype=np.float64)


# ─── Synthetic stream for demo mode ─────────────────────────
class DemoEMGStream(BaseEMGStream):
    def __init__(self) -> None:
        from src.stream import EMGSample, FrameStats
        self._running = False
        self._stats = FrameStats()
        self._activity_idx = 0
        self._activities = ["typing", "typing", "mouse_use", "idle", "stretching"]
        self._switch_interval = 30.0
        self._start_time = time.perf_counter()

    def start(self) -> None:
        self._running = True
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        self._running = False

    def consume_samples(self, max_items: int = 256) -> list:
        from src.stream import EMGSample
        if not self._running:
            return []
        elapsed = time.perf_counter() - self._start_time
        self._activity_idx = int(elapsed / self._switch_interval) % len(self._activities)
        activity = self._activities[self._activity_idx]
        samples = []
        now = time.perf_counter()
        for i in range(50):
            values = self._generate_emg(activity, now + i * 0.001)
            samples.append(EMGSample(timestamp=now + i * 0.001, values=values))
        self._stats.total_frames += len(samples)
        self._stats.emg_frames += len(samples)
        return samples

    def _generate_emg(self, activity: str, t: float) -> np.ndarray:
        rng = np.random.default_rng(int(t * 1000) % (2**31))
        base = rng.normal(0, 1, CHANNEL_COUNT)
        if activity == "typing":
            base[:4] *= 50; base[4:] *= 10
            for i in range(4):
                base[i] += 30 * np.sin(2 * np.pi * 3 * t + i)
        elif activity == "mouse_use":
            base[:2] *= 40; base[2:] *= 5
        elif activity == "stretching":
            base *= 60
            for i in range(CHANNEL_COUNT):
                base[i] += 20 * np.sin(2 * np.pi * 0.5 * t + i * 0.3)
        else:
            base *= 3
        return base

    def consume_imu(self, max_items: int = 256) -> list:
        return []

    def frame_stats(self):
        return self._stats


# ─── Demo classifier ────────────────────────────────────────
class DemoClassifier:
    def __init__(self, stream_ref):
        self._stream = stream_ref
        self.classes = ["typing", "mouse_use", "idle", "stretching"]
        self.n_features = 84

    def predict(self, features: np.ndarray) -> tuple:
        idx = 0
        if self._stream and isinstance(self._stream, DemoEMGStream):
            idx = self._stream._activity_idx % len(self.classes)
        probs = {c: 0.05 for c in self.classes}
        probs[self.classes[idx]] = 0.85
        return self.classes[idx], 0.85, probs


# ─── Routes ──────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def status():
    return {
        "connected": stream is not None,
        "model_loaded": onnx_session is not None,
        "demo_mode": demo_mode,
        "timeline_entries": len(timeline),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_connections.add(ws)
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            if data.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        active_connections.discard(ws)
    except Exception:
        active_connections.discard(ws)


async def broadcast(data: dict):
    dead = set()
    for ws in active_connections:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    active_connections.difference_update(dead)


# ─── Main processing loop ───────────────────────────────────
async def processing_loop():
    global fatigue_estimator, decision_engine

    window_samples = int(WINDOW_SEC * SAMPLE_RATE)
    fatigue_estimator = FatigueEstimator(sample_rate=SAMPLE_RATE)
    decision_engine = DecisionEngine()

    emg_buffer = np.zeros((0, CHANNEL_COUNT))
    last_push = time.time()
    total_received = 0
    log_interval = time.time()

    while True:
        if stream is None:
            await asyncio.sleep(0.1)
            continue

        samples = stream.consume_samples(max_items=512)
        if not samples:
            await asyncio.sleep(0.02)
            continue

        total_received += len(samples)
        new_data = np.array([s.values for s in samples])
        emg_buffer = np.vstack([emg_buffer, new_data]) if emg_buffer.size else new_data

        if emg_buffer.shape[0] > SAMPLE_RATE * 10:
            emg_buffer = emg_buffer[-SAMPLE_RATE * 5:]

        now = time.time()
        if now - log_interval >= 5.0:
            stats = stream.frame_stats()
            rms_now = np.sqrt(np.mean(new_data ** 2, axis=0))
            print(f"[data] total={total_received}, buffer={emg_buffer.shape[0]}, "
                  f"emg_frames={stats.emg_frames}, dropped={stats.dropped_frames}, "
                  f"rms_ch1={rms_now[0]:.1f}")
            log_interval = now

        if now - last_push < 0.2:
            await asyncio.sleep(0.01)
            continue
        last_push = now

        activity = "idle"
        confidence = 0.0
        probabilities = {}

        if onnx_session is not None and emg_buffer.shape[0] >= window_samples:
            window = emg_buffer[-window_samples:]
            try:
                feats = extract_window_features_84(window, SAMPLE_RATE)
                activity, confidence, probabilities = onnx_session.predict(feats)
            except Exception as e:
                print(f"[inference] {e}")

        fatigue_reading = FatigueReading(0.0, 0.0, 0.0, 0.0, 0.0)
        if fatigue_estimator and emg_buffer.shape[0] >= window_samples:
            try:
                window = emg_buffer[-window_samples:]
                fatigue_reading = fatigue_estimator.update(now, window)
            except Exception:
                pass

        decision = None
        if decision_engine:
            try:
                decision = decision_engine.update(activity, fatigue_reading.score, now)
            except Exception:
                pass

        rms_values = np.sqrt(np.mean(emg_buffer[-min(250, len(emg_buffer)):] ** 2, axis=0)).tolist()

        payload = {
            "type": "state_update",
            "timestamp": now,
            "activity": activity,
            "confidence": confidence,
            "probabilities": probabilities,
            "fatigue": {
                "score": fatigue_reading.score,
                "mdf_current": fatigue_reading.mdf_current,
                "mdf_baseline": fatigue_reading.mdf_baseline,
                "mdf_slope": fatigue_reading.mdf_slope,
                "confidence": fatigue_reading.confidence,
            },
            "rms": rms_values,
            "emg_sample_count": len(emg_buffer),
        }

        if decision:
            payload["decision"] = {
                "state": decision.state.value,
                "recommendation": decision.recommendation.value,
                "urgency": decision.urgency,
                "reasons": decision.reasons,
                "continuous_work_min": decision.continuous_work_min,
                "total_work_min": decision.total_work_min,
            }
            if decision.urgency >= 0.5:
                timeline.append({
                    "time": now,
                    "type": "alert",
                    "message": decision.reasons[0] if decision.reasons else "",
                    "urgency": decision.urgency,
                })
                if len(timeline) > 200:
                    timeline[:] = timeline[-100:]

        await broadcast(payload)
        await asyncio.sleep(0.01)


@app.on_event("startup")
async def startup():
    asyncio.create_task(processing_loop())


# ─── Entrypoint ──────────────────────────────────────────────
def main():
    global stream, onnx_session, onnx_classes, onnx_config, demo_mode

    parser = argparse.ArgumentParser(description="FluxChi Web Server")
    parser.add_argument("--port", help="Serial port for WAVELETECH wristband")
    parser.add_argument("--demo", action="store_true", help="Demo mode (no hardware)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8000)
    parser.add_argument("--model-dir", default="model")
    args = parser.parse_args()

    demo_mode = args.demo

    model_dir = Path(args.model_dir)
    for prefix in ("activity_classifier", "ninapro_classifier"):
        onnx_path = model_dir / f"{prefix}.onnx"
        config_path = model_dir / f"{prefix}_config.json"
        if onnx_path.exists() and config_path.exists():
            config = json.loads(config_path.read_text())
            try:
                onnx_session = ONNXClassifier(str(onnx_path), config)
                onnx_config = config
                onnx_classes = config.get("classes", [])
                print(f"[web] Loaded ONNX model: {onnx_path}")
                print(f"[web]   Classes: {onnx_classes}")
                print(f"[web]   Features: {config.get('n_features', '?')}")
                print(f"[web]   Accuracy: {config.get('accuracy', '?')}")
                break
            except Exception as e:
                print(f"[web] Failed to load {onnx_path}: {e}")

    if demo_mode:
        print("[web] Running in DEMO mode (synthetic data)")
        stream = DemoEMGStream()
        stream.start()
        if onnx_session is None:
            onnx_session = DemoClassifier(stream)
            print("[web] Using demo classifier")
    elif args.port:
        print(f"[web] Connecting to {args.port}...")
        stream = SerialEMGStream(args.port)
        stream.start()
    else:
        print("[web] No port specified and --demo not set.")
        print("[web] Use --port /dev/tty.usbserial-XXXX or --demo")
        sys.exit(1)

    if onnx_session is None:
        print("[web] No model loaded — signal-only mode (EMG + fatigue, no classification)")

    import uvicorn
    print(f"\n  FluxChi Web: http://localhost:{args.web_port}")
    print(f"  Demo mode: {demo_mode}")
    print(f"  Model: {'ONNX loaded' if onnx_session else 'none'}\n")
    uvicorn.run(app, host=args.host, port=args.web_port, log_level="warning")


if __name__ == "__main__":
    main()
