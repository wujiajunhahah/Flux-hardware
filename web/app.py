"""
FluxChi — API Server
=========================================================
RESTful API + SSE + WebSocket 三种接入方式。

  REST   — iOS / 第三方轮询       /api/v1/*
  SSE    — iOS URLSession 实时流  /api/v1/stream
  WS     — Web 仪表盘实时推送      /ws

用法：
  python web/app.py --port /dev/tty.usbserial-XXXX
  python web/app.py --demo
  python web/app.py --demo --speed 10
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

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.stream import CHANNEL_COUNT, BaseEMGStream, BleEMGStream, SerialEMGStream
from src.energy import StaminaEngine, StaminaState
from src.decision import DecisionEngine

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="FluxChi",
    version="2.1",
    description="EMG-based work sustainability API for iOS, Web, and beyond.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── Global state ────────────────────────────────────────────
stream: Optional[BaseEMGStream] = None
onnx_session = None
onnx_classes: List[str] = []
onnx_config: Optional[dict] = None
stamina_engine: Optional[StaminaEngine] = None
decision_engine: Optional[DecisionEngine] = None
active_connections: Set[WebSocket] = set()
demo_mode = False
speed_multiplier = 1.0
timeline: List[Dict] = []
latest_payload: Dict = {}
_server_start: float = time.time()
SAMPLE_RATE = 1000
WINDOW_SEC = 0.25


def _ok(data: dict) -> dict:
    return {"ok": True, "ts": time.time(), "data": data}


def _err(code: str, message: str, status: int = 200) -> dict:
    return {"ok": False, "ts": time.time(), "error": code, "message": message}


# ─── ONNX Inference wrapper ─────────────────────────────────
class ONNXClassifier:
    def __init__(self, onnx_path: str, config: dict):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(onnx_path))
        self.input_name = self.session.get_inputs()[0].name
        self.classes = config.get("classes", [])
        self.n_features = config.get("n_features", 84)

    def predict(self, features: np.ndarray) -> tuple:
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


# ─── Feature extraction ─────────────────────────────────────
def extract_window_features_84(window: np.ndarray, sample_rate: int) -> np.ndarray:
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


# ─── Demo streams ───────────────────────────────────────────
class DemoEMGStream(BaseEMGStream):
    def __init__(self) -> None:
        from src.stream import FrameStats
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  API v1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Dashboard ────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── System ───────────────────────────────────────────────────
@app.get("/api/v1/status", tags=["System"], summary="服务状态")
async def v1_status():
    """返回服务连接、模型、模式等元信息。"""
    return _ok({
        "connected": stream is not None,
        "model_loaded": onnx_session is not None,
        "demo_mode": demo_mode,
        "speed": speed_multiplier,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNEL_COUNT,
        "uptime_sec": round(time.time() - _server_start, 1),
        "websocket_clients": len(active_connections),
    })


@app.get("/api/v1/model", tags=["System"], summary="模型信息")
async def v1_model():
    """返回已加载 ONNX 模型的类别和精度。"""
    if onnx_config:
        return _ok({
            "loaded": True,
            "classes": onnx_config.get("classes", []),
            "n_features": onnx_config.get("n_features"),
            "accuracy": onnx_config.get("accuracy"),
        })
    return _ok({"loaded": False, "classes": []})


# ── Core — Stamina ───────────────────────────────────────────
@app.get("/api/v1/pulse", tags=["Core"], summary="轻量心跳（iOS 后台用）")
async def v1_pulse():
    """最小化 payload：仅 stamina + state + recommendation。适合 iOS Background Task 每 30s 轮询。"""
    if not latest_payload:
        return _err("no_data", "传感器数据尚未就绪")
    st = latest_payload.get("stamina", {})
    dec = latest_payload.get("decision", {})
    return _ok({
        "stamina": st.get("value", 0),
        "state": st.get("state", "unknown"),
        "recommendation": dec.get("recommendation", "unknown"),
        "urgency": dec.get("urgency", 0),
        "suggested_work_min": st.get("suggested_work_min", 0),
        "suggested_break_min": st.get("suggested_break_min", 0),
    })


@app.get("/api/v1/stamina", tags=["Core"], summary="续航详情")
async def v1_stamina():
    """完整续航分数 + 三维度指标 + 建议时长。"""
    if stamina_engine is None or stamina_engine.latest is None:
        return _err("no_data", "StaminaEngine 尚未初始化")
    r = stamina_engine.latest
    return _ok({
        "stamina": r.stamina,
        "state": r.state.value,
        "dimensions": {
            "consistency": r.consistency,
            "tension": r.tension,
            "fatigue": r.fatigue,
        },
        "rates": {
            "drain_per_min": r.drain_rate,
            "recovery_per_min": r.recovery_rate,
        },
        "suggestion": {
            "work_min": r.suggested_work_min,
            "break_min": r.suggested_break_min,
        },
        "session": {
            "continuous_work_min": r.continuous_work_min,
            "total_work_min": r.total_work_min,
        },
    })


@app.get("/api/v1/decision", tags=["Core"], summary="决策建议")
async def v1_decision():
    """当前工作/休息建议 + 原因列表。"""
    dec = latest_payload.get("decision")
    if not dec:
        return _err("no_data", "尚无决策数据")
    return _ok(dec)


@app.get("/api/v1/state", tags=["Core"], summary="完整快照")
async def v1_state():
    """返回与 WebSocket / SSE 推送完全相同的完整 payload。"""
    if not latest_payload:
        return _err("no_data", "传感器数据尚未就绪")
    return _ok(latest_payload)


# ── Signal ───────────────────────────────────────────────────
@app.get("/api/v1/emg", tags=["Signal"], summary="EMG 通道 RMS")
async def v1_emg():
    """8 通道实时 RMS 值。"""
    rms = latest_payload.get("rms", [0.0] * CHANNEL_COUNT)
    return _ok({
        "channels": CHANNEL_COUNT,
        "rms": rms,
        "sample_count": latest_payload.get("emg_sample_count", 0),
    })


# ── History ──────────────────────────────────────────────────
@app.get("/api/v1/timeline", tags=["History"], summary="提醒历史")
async def v1_timeline(
    limit: int = Query(50, ge=1, le=200, description="返回最近 N 条"),
):
    """最近的休息提醒历史。"""
    return _ok({
        "total": len(timeline),
        "events": timeline[-limit:],
    })


# ── Control ──────────────────────────────────────────────────
@app.post("/api/v1/stamina/reset", tags=["Control"], summary="重置续航")
async def v1_stamina_reset():
    """将续航重置为 100%。"""
    if stamina_engine:
        stamina_engine.reset()
        return _ok({"stamina": 100.0})
    return _err("not_ready", "StaminaEngine 未初始化")


@app.post("/api/v1/stamina/save", tags=["Control"], summary="持久化状态")
async def v1_stamina_save():
    """保存当前续航到磁盘，下次启动可恢复。"""
    if stamina_engine:
        stamina_engine.save()
        return _ok({"stamina": stamina_engine.stamina})
    return _err("not_ready", "StaminaEngine 未初始化")


@app.post("/api/v1/stamina/load", tags=["Control"], summary="加载状态")
async def v1_stamina_load():
    """从磁盘恢复上次保存的续航。"""
    if stamina_engine:
        loaded = stamina_engine.load()
        return _ok({"loaded": loaded, "stamina": stamina_engine.stamina})
    return _err("not_ready", "StaminaEngine 未初始化")


# ── SSE Stream (iOS-friendly) ────────────────────────────────
@app.get("/api/v1/stream", tags=["Realtime"], summary="SSE 实时流")
async def v1_sse_stream():
    """
    Server-Sent Events 流。比 WebSocket 更适合 iOS URLSession。

    每 ~200ms 推送一帧，格式：
        event: state
        data: { ... payload JSON ... }
    """
    async def event_generator():
        last_ts = 0.0
        while True:
            await asyncio.sleep(0.2)
            if not latest_payload:
                continue
            ts = latest_payload.get("timestamp", 0)
            if ts == last_ts:
                continue
            last_ts = ts
            payload_json = json.dumps(latest_payload, ensure_ascii=False)
            yield f"event: state\ndata: {payload_json}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Legacy (backward compat, redirect-free) ──────────────────
@app.get("/api/status", include_in_schema=False)
async def legacy_status():
    return await v1_status()

@app.get("/api/stamina", include_in_schema=False)
async def legacy_stamina():
    return await v1_stamina()

@app.get("/api/state", include_in_schema=False)
async def legacy_state():
    return await v1_state()

@app.get("/api/decision", include_in_schema=False)
async def legacy_decision():
    return await v1_decision()

@app.get("/api/emg", include_in_schema=False)
async def legacy_emg():
    return await v1_emg()

@app.get("/api/model", include_in_schema=False)
async def legacy_model():
    return await v1_model()

@app.get("/api/timeline", include_in_schema=False)
async def legacy_timeline():
    return await v1_timeline()

@app.post("/api/stamina/reset", include_in_schema=False)
async def legacy_reset():
    return await v1_stamina_reset()

@app.post("/api/stamina/save", include_in_schema=False)
async def legacy_save():
    return await v1_stamina_save()

@app.post("/api/stamina/load", include_in_schema=False)
async def legacy_load():
    return await v1_stamina_load()


# ── WebSocket (Web dashboard) ────────────────────────────────
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


# ─── Processing loop ────────────────────────────────────────
async def processing_loop():
    global stamina_engine, decision_engine

    window_samples = int(WINDOW_SEC * SAMPLE_RATE)
    stamina_engine = StaminaEngine(sample_rate=SAMPLE_RATE, speed=speed_multiplier)
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

        stamina_reading = None
        if stamina_engine and emg_buffer.shape[0] >= window_samples:
            try:
                window = emg_buffer[-window_samples:]
                stamina_reading = stamina_engine.update(window, activity, now)
            except Exception as e:
                print(f"[stamina] {e}")

        decision = None
        if decision_engine and stamina_reading:
            try:
                decision = decision_engine.update(stamina_reading, activity)
            except Exception as e:
                print(f"[decision] {e}")

        rms_values = np.sqrt(np.mean(emg_buffer[-min(250, len(emg_buffer)):] ** 2, axis=0)).tolist()

        payload = {
            "type": "state_update",
            "timestamp": now,
            "activity": activity,
            "confidence": confidence,
            "probabilities": probabilities,
            "rms": rms_values,
            "emg_sample_count": len(emg_buffer),
        }

        if stamina_reading:
            payload["stamina"] = {
                "value": stamina_reading.stamina,
                "state": stamina_reading.state.value,
                "consistency": stamina_reading.consistency,
                "tension": stamina_reading.tension,
                "fatigue": stamina_reading.fatigue,
                "drain_rate": stamina_reading.drain_rate,
                "recovery_rate": stamina_reading.recovery_rate,
                "suggested_work_min": stamina_reading.suggested_work_min,
                "suggested_break_min": stamina_reading.suggested_break_min,
                "continuous_work_min": stamina_reading.continuous_work_min,
                "total_work_min": stamina_reading.total_work_min,
            }

        if decision:
            payload["decision"] = {
                "state": decision.state,
                "recommendation": decision.recommendation.value,
                "urgency": decision.urgency,
                "reasons": decision.reasons,
                "stamina": decision.stamina,
                "continuous_work_min": decision.continuous_work_min,
                "total_work_min": decision.total_work_min,
                "suggested_work_min": decision.suggested_work_min,
                "suggested_break_min": decision.suggested_break_min,
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

        latest_payload.clear()
        latest_payload.update(payload)

        await broadcast(payload)
        await asyncio.sleep(0.01)


@app.on_event("startup")
async def startup():
    asyncio.create_task(processing_loop())


# ─── Entrypoint ──────────────────────────────────────────────
def main():
    global stream, onnx_session, onnx_classes, onnx_config, demo_mode, speed_multiplier

    parser = argparse.ArgumentParser(description="FluxChi API Server")
    parser.add_argument("--port", help="Serial port for WAVELETECH wristband")
    parser.add_argument("--ble", nargs="?", const="auto", default=None,
                        help="Connect via BLE (no USB dongle). Optionally pass device address.")
    parser.add_argument("--demo", action="store_true", help="Demo mode (synthetic data)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8000)
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Stamina speed multiplier (default=1, defense demo=10)")
    args = parser.parse_args()

    demo_mode = args.demo
    speed_multiplier = args.speed

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
        print("[web] DEMO mode (synthetic data)")
        stream = DemoEMGStream()
        stream.start()
        if onnx_session is None:
            onnx_session = DemoClassifier(stream)
            print("[web] Using demo classifier")
    elif args.ble is not None:
        addr = None if args.ble == "auto" else args.ble
        print(f"[web] BLE mode — {'auto-scan' if addr is None else addr}")
        stream = BleEMGStream(address=addr)
        stream.start()
    elif args.port:
        print(f"[web] Connecting to {args.port}...")
        stream = SerialEMGStream(args.port)
        stream.start()
    else:
        print("[web] No port specified and --demo not set.")
        print("[web] Use --port /dev/tty.usbserial-XXXX  or  --ble  or  --demo")
        sys.exit(1)

    if onnx_session is None:
        print("[web] No model — signal-only mode")

    import uvicorn
    p = args.web_port
    print(f"\n  FluxChi API Server")
    print(f"  ─────────────────────────────")
    print(f"  Dashboard   http://localhost:{p}")
    print(f"  Swagger     http://localhost:{p}/docs")
    print(f"  SSE stream  http://localhost:{p}/api/v1/stream")
    print(f"  Pulse       http://localhost:{p}/api/v1/pulse")
    mode = "demo" if demo_mode else ("ble" if args.ble is not None else "serial")
    print(f"  Mode: {mode}  Speed: {speed_multiplier}x\n")
    uvicorn.run(app, host=args.host, port=p, log_level="warning")


if __name__ == "__main__":
    main()
