"""
FluxChi — API Server
=========================================================
RESTful API + SSE + WebSocket 三种接入方式。

  REST   — iOS / 第三方轮询       /api/v1/*
  SSE    — iOS URLSession 实时流  /api/v1/stream
  WS     — Web 仪表盘实时推送      /ws
  WS     — 浏览器视觉特征流        /ws/vision  (见 docs/VISION-WEBSOCKET-SPEC.md)

用法：
  python web/app.py --port /dev/tty.usbserial-XXXX
  python web/app.py --demo
  python web/app.py --demo --speed 10
  python web/app.py --web-port 8000
      无 --demo/--port/--ble 时数据源为「空闲」，在网页「数据源」或 POST /api/v1/transport/apply 切换
=========================================================
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Body, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.stream import (
    CHANNEL_COUNT,
    BaseEMGStream,
    BleEMGStream,
    SerialEMGStream,
    normalize_serial_device,
)
from src.canonical_signal import CANONICAL_EMG_CHANNELS, canonicalize_export_package, normalize_rms_to_canonical
from src.features import remove_dc
from src.session_insight import session_insight_text
from src.session_store import list_sessions_meta, load_session, save_session_package, session_file_path
from src.energy import StaminaEngine, StaminaState
from src.decision import DecisionEngine
from src.fusion_engine import FusionEngine
from src.vision_engine import VisionEngine, vision_reading_alerts
from src.sensor_module import ModuleReading
from src.modules.emg_module import EmgModule
from src.modules.vision_module import VisionModule
from src.context_bus import ContextBus

STATIC_DIR = Path(__file__).parent / "static"

@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    _ensure_vision_stack()
    task = asyncio.create_task(processing_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="FluxChi",
    version="2.1",
    description="EMG-based work sustainability API for iOS, Web, and beyond.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=_app_lifespan,
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
_real_onnx_session = None  # 文件加载的 ONNX，切换 demo 时不丢失
onnx_classes: List[str] = []
onnx_config: Optional[dict] = None
stamina_engine: Optional[StaminaEngine] = None
decision_engine: Optional[DecisionEngine] = None
active_connections: Set[WebSocket] = set()
vision_engine: Optional[VisionEngine] = None
fusion_engine: Optional[FusionEngine] = None
emg_module: Optional[EmgModule] = None
vision_module: Optional[VisionModule] = None
context_bus: Optional[ContextBus] = None
vision_ws_connections: Set[WebSocket] = set()
_last_vision_frame_ts: float = 0.0
VISION_FRAME_STALE_SEC = 2.5
demo_mode = False
speed_multiplier = 1.0
timeline: List[Dict] = []
latest_payload: Dict = {}
_server_start: float = time.time()
SAMPLE_RATE = 1000
WINDOW_SEC = 0.25

_transport_lock = asyncio.Lock()
_emg_buffer_reset_requested = False
current_serial_port: Optional[str] = None
current_ble_address: Optional[str] = None

# 网页极简录制：由 processing_loop 按 500ms 写入快照（与 iOS snapshotInterval 一致）
web_recording: Optional[Dict[str, Any]] = None


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
        raw0 = outputs[0]
        flat0 = np.asarray(raw0).flatten()
        first = flat0[0] if flat0.size else 0
        label: str
        if self.classes and isinstance(first, (np.integer, int, np.int64, np.int32)):
            idx = int(first)
            if 0 <= idx < len(self.classes):
                label = str(self.classes[idx])
            else:
                label = str(idx)
        elif self.classes and isinstance(first, (float, np.floating)) and float(first) == int(first):
            idx = int(first)
            if 0 <= idx < len(self.classes):
                label = str(self.classes[idx])
            else:
                label = str(first)
        else:
            label = str(first)
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
    window = remove_dc(np.asarray(window, dtype=np.float64))
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


def _estimate_eff_sample_rate(n_rows: int, ts: np.ndarray) -> float:
    """用 perf_counter 时间戳估计有效采样率 (Hz)。"""
    if n_rows < 2 or ts.shape[0] != n_rows:
        return float(SAMPLE_RATE)
    span = float(ts[-1] - ts[0])
    if span <= 1e-9:
        return float(SAMPLE_RATE)
    eff = (n_rows - 1) / span
    return float(np.clip(eff, 50.0, 4000.0))


def _emg_ac_quality_from_buffer(emg_buffer: np.ndarray) -> float:
    """多通道 AC-RMS 平衡启发式 → 0–1，供 fusion quality。"""
    if emg_buffer.shape[0] < 16:
        return 0.45
    seg = remove_dc(np.asarray(emg_buffer[-min(500, len(emg_buffer)):], dtype=np.float64))
    rms_c = np.sqrt(np.mean(seg**2, axis=0))
    mean_r = float(np.mean(rms_c))
    if mean_r < 30.0:
        return 0.2
    cv = float(np.std(rms_c) / (mean_r + 1e-9))
    q = 1.0 - min(cv, 1.2) / 1.2
    return float(np.clip(q, 0.15, 1.0))


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


def _raw_serial_port_rows() -> List[Dict[str, str]]:
    try:
        from serial.tools import list_ports

        rows: List[Dict[str, str]] = []
        for p in list_ports.comports():
            rows.append(
                {
                    "device": p.device,
                    "description": (p.description or "").strip(),
                    "hwid": (getattr(p, "hwid", None) or "").strip(),
                }
            )
        return rows
    except Exception as exc:
        print(f"[transport] 枚举串口失败（常因未安装 pyserial：pip install pyserial）: {exc}")
        return []


def _dedupe_mac_tty_vs_cu(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if sys.platform != "darwin":
        return rows
    devs = {r["device"] for r in rows}
    out: List[Dict[str, str]] = []
    for r in rows:
        dev = r["device"]
        if dev.startswith("/dev/tty."):
            cu = "/dev/cu." + dev[len("/dev/tty.") :]
            if cu in devs:
                continue
        out.append(r)
    return out


def _score_serial_row(row: Dict[str, str]) -> int:
    d = row["device"].lower()
    desc = row["description"].lower()
    hw = row.get("hwid", "").upper()
    s = 0
    if "usbserial" in d or "usbserial" in desc:
        s += 55
    if "usbmodem" in d and "bluetooth" not in d:
        s += 22
    for k in (
        "cp210",
        "ch340",
        "ch341",
        "ftdi",
        "cp2102",
        "uart bridge",
        "usb uart",
        "usb-to-uart",
        "serial adapter",
    ):
        if k in desc:
            s += 28
    for vid in ("10C4", "1A86", "0403"):
        if vid in hw:
            s += 18
    if d.startswith("/dev/cu."):
        s += 8
    if d.startswith("/dev/tty.") and sys.platform == "darwin":
        s -= 12
    for bad in ("bluetooth-incoming", "debug-console"):
        if bad in d:
            s -= 95
    if "bluetooth" in d or "bluetooth" in desc:
        s -= 32
    return s


def _serialize_port_for_api(row: Dict[str, str], score: int) -> Dict[str, object]:
    return {
        "device": normalize_serial_device(row["device"]),
        "description": row["description"] or "n/a",
        "likely_emg_bridge": score >= 22,
    }


def _build_serial_catalog(include_other: bool) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    raw = _dedupe_mac_tty_vs_cu(_raw_serial_port_rows())
    scored = sorted(((r, _score_serial_row(r)) for r in raw), key=lambda t: -t[1])
    items = [(_serialize_port_for_api(r, sc), sc) for r, sc in scored]
    strong = [p for p, sc in items if sc >= 22]
    weak = [p for p, sc in items if sc < 22]
    if not strong:
        primary = [p for p, _ in items]
        secondary: List[Dict[str, object]] = []
    else:
        primary = strong
        secondary = weak if include_other else []
    return primary, secondary


def _resolve_serial_port(explicit: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """显式路径会规范为 macOS 的 cu.*；自动选择时优先 USB 串口桥得分高的设备。"""
    ex = (explicit or "").strip()
    if ex:
        return normalize_serial_device(ex), None

    raw = _dedupe_mac_tty_vs_cu(_raw_serial_port_rows())
    if not raw:
        return None, "未检测到串口，请插上 USB 接收器或在「手动路径」填写路径"

    scored = sorted(((r, _score_serial_row(r)) for r in raw), key=lambda t: -t[1])
    strong = [(r, sc) for r, sc in scored if sc >= 22]
    pool: List[tuple[Dict[str, str], int]] = strong if strong else scored

    if len(pool) == 1:
        return normalize_serial_device(pool[0][0]["device"]), None

    top0, sc0 = pool[0][0], pool[0][1]
    sc1 = pool[1][1] if len(pool) > 1 else -999
    if sc0 >= 40 and (sc0 - sc1) >= 18:
        return normalize_serial_device(top0["device"]), None

    preview = "、".join(normalize_serial_device(r["device"]) for r, _ in pool[:5])
    return None, (
        "无法自动唯一选定接收器。请在下拉中选 likely_emg_bridge 或含 CP2102/usbserial 的 /dev/cu.*；"
        "若已连接仍无数据，请确认手环在发送数据。候选："
        f"{preview}"
    )


def _serial_apply_message(port: str, auto: bool) -> str:
    if auto:
        return f"已自动连接串口：{port}"
    return f"已连接串口 {port}"


def _stream_signal_stats() -> Optional[Dict[str, object]]:
    if stream is None:
        return None
    try:
        st = stream.frame_stats()
        return {
            "emg_frames": st.emg_frames,
            "total_frames": st.total_frames,
            "imu_frames": st.imu_frames,
            "dropped_frames": st.dropped_frames,
        }
    except Exception:
        return None


def _vision_signal_summary() -> Dict[str, Any]:
    """供 /api/v1/transport 与 status：摄像头流与手环数据源独立。"""
    now = time.time()
    n = len(vision_ws_connections)
    if _last_vision_frame_ts <= 0:
        age = None
        receiving = False
    else:
        age = round(now - _last_vision_frame_ts, 2)
        receiving = (now - _last_vision_frame_ts) < VISION_FRAME_STALE_SEC
    return {
        "ws_clients": n,
        "receiving": receiving,
        "last_frame_age_sec": age,
    }


def _transport_mode_label() -> str:
    if stream is None:
        return "idle"
    if isinstance(stream, DemoEMGStream):
        return "demo"
    if isinstance(stream, SerialEMGStream):
        return "serial"
    if isinstance(stream, BleEMGStream):
        return "ble"
    return "unknown"


def _request_emg_buffer_reset() -> None:
    global _emg_buffer_reset_requested
    _emg_buffer_reset_requested = True


class TransportApplyBody(BaseModel):
    mode: str = Field(..., description="idle | demo | serial | ble")
    port: Optional[str] = Field(
        None,
        description="serial 模式：串口路径；省略时若本机仅 1 个串口则自动选用",
    )
    address: Optional[str] = Field(None, description="ble 模式：设备地址，留空则自动扫描")


def _capture_web_recording_snapshot() -> None:
    """processing_loop 在写入 latest_payload 后调用；500ms 节流。"""
    global web_recording
    if not web_recording or not latest_payload:
        return
    now = time.time()
    if now - web_recording["_last_snap"] < 0.5:
        return
    web_recording["_last_snap"] = now
    t0 = float(web_recording["started_at"])
    st = latest_payload.get("stamina") or {}
    rms = latest_payload.get("rms") or []
    n_src = len(rms) if isinstance(rms, list) else 0
    r8 = normalize_rms_to_canonical(rms if isinstance(rms, list) else [], source_channels=n_src or CHANNEL_COUNT)
    web_recording["snapshots"].append(
        {
            "t": round(now - t0, 3),
            "segIdx": 0,
            "stamina": round(float(st.get("value", 0)), 1),
            "state": st.get("state", "unknown"),
            "con": round(float(st.get("consistency", 0)), 3),
            "ten": round(float(st.get("tension", 0)), 3),
            "fat": round(float(st.get("fatigue", 0)), 3),
            "act": latest_payload.get("activity", "idle"),
            "conf": round(float(latest_payload.get("confidence", 0)), 2),
            "rms": r8,
        }
    )


def _build_web_session_package(rec: Dict[str, Any]) -> Dict[str, Any]:
    import uuid as _uuid

    snaps = rec.get("snapshots") or []
    sid = str(_uuid.uuid4())
    dur = float(snaps[-1]["t"]) if snaps else 0.0
    iso_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(rec["started_at"]))
    iso_end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "fluxchiVersion": "2.1-web",
        "schemaVersion": 1,
        "exportedAt": iso_end,
        "device": {"model": "FluxChi-Web", "systemVersion": "", "source": "wifi"},
        "session": {
            "id": sid,
            "startedAt": iso_start,
            "endedAt": iso_end,
            "title": "Web 记录",
            "source": "wifi",
            "durationSec": round(dur, 1),
        },
        "signalMeta": {
            "totalChannels": CANONICAL_EMG_CHANNELS,
            "activeChannels": CHANNEL_COUNT,
            "sampleIntervalMs": 500,
            "snapshotCount": len(snaps),
            "note": "网关网页录制；与 iOS ExportPackage 同构，已由后端规范 8 路 RMS。",
        },
        "segments": [
            {
                "id": str(_uuid.uuid4()),
                "label": "work",
                "startedAt": iso_start,
                "endedAt": iso_end,
                "durationSec": round(dur, 1),
                "snapshotCount": len(snaps),
            }
        ],
        "snapshots": list(snaps),
        "channelStats": [],
        "feedback": None,
        "summary": None,
        "processing": {
            "staminaWeights": {"consistency": 0.40, "tension": 0.25, "fatigue": 0.35},
            "staminaThresholds": {"focused": 60, "fading": 30, "depleted": 0},
            "emgDecoding": "gateway_web_recorder",
            "featureWindow": "aligned_with_ios_export",
        },
    }


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
        "transport_mode": _transport_mode_label(),
        "serial_port": current_serial_port,
        "ble_address": current_ble_address,
        "speed": speed_multiplier,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNEL_COUNT,
        "uptime_sec": round(time.time() - _server_start, 1),
        "websocket_clients": len(active_connections),
        "vision": _vision_signal_summary(),
    })


@app.get("/api/v1/transport", tags=["System"], summary="数据源状态与串口列表")
async def v1_transport(
    include_all: bool = Query(
        False,
        description="为 true 时附带 serial_ports_other（耳机/蓝牙等低优先级端口）",
    ),
):
    """
    `serial_ports` 为筛选、排序后的列表（优先 USB 串口桥）；`stream_stats` 用于判断是否真的收到 EMG 帧。
    """
    primary, secondary = _build_serial_catalog(include_other=include_all)
    data: Dict[str, object] = {
        "mode": _transport_mode_label(),
        "demo_mode": demo_mode,
        "serial_port": current_serial_port,
        "ble_address": current_ble_address,
        "serial_ports": primary,
        "signal_hint": (
            "若已选串口但 emg_frames 长时间为 0：多半选错设备；macOS 请用 /dev/cu.*；"
            "并确认手环已通过该接收器以 921600 波特发送 D2D2D2 协议包。"
        ),
    }
    if include_all and secondary:
        data["serial_ports_other"] = secondary
    stats = _stream_signal_stats()
    if stats is not None:
        data["stream_stats"] = stats
    data["vision"] = _vision_signal_summary()
    data["sources_note"] = "手环（EMG）与摄像头（/ws/vision）可同时启用，互不影响。"
    return _ok(data)


@app.post("/api/v1/transport/apply", tags=["System"], summary="切换数据源")
async def v1_transport_apply(body: TransportApplyBody):
    """运行时切换演示数据 / USB 串口 / BLE / 断开。与命令行 `--demo` `--port` `--ble` 等价。"""
    global stream, onnx_session, demo_mode, latest_payload
    global current_serial_port, current_ble_address

    mode = (body.mode or "").strip().lower()
    if mode not in ("idle", "demo", "serial", "ble"):
        return _err("bad_request", f"未知 mode: {body.mode!r}")

    async with _transport_lock:
        _request_emg_buffer_reset()

        old = stream
        stream = None
        if old is not None:
            try:
                old.stop()
            except Exception as exc:
                print(f"[transport] stop previous stream: {exc}")

        demo_mode = False
        current_serial_port = None
        current_ble_address = None

        if mode == "idle":
            latest_payload.clear()
            onnx_session = _real_onnx_session
            return _ok({"mode": "idle", "message": "已断开数据源"})

        if mode == "demo":
            demo_mode = True
            new_stream = DemoEMGStream()
            new_stream.start()
            stream = new_stream
            onnx_session = _real_onnx_session if _real_onnx_session is not None else DemoClassifier(stream)
            return _ok({"mode": "demo", "message": "已切换为演示数据"})

        if mode == "serial":
            auto_pick = not (body.port or "").strip()
            port, res_err = _resolve_serial_port(body.port)
            if res_err or not port:
                return _err("bad_request", res_err or "无法解析串口")
            try:
                new_stream = SerialEMGStream(port)
                new_stream.start()
            except Exception as exc:
                return _err("open_failed", str(exc))
            stream = new_stream
            current_serial_port = port
            onnx_session = _real_onnx_session
            return _ok(
                {
                    "mode": "serial",
                    "serial_port": port,
                    "auto_selected": auto_pick,
                    "message": _serial_apply_message(port, auto_pick),
                }
            )

        # ble
        addr = (body.address or "").strip() or None
        try:
            new_stream = BleEMGStream(address=addr)
            new_stream.start()
        except Exception as exc:
            return _err("open_failed", str(exc))
        stream = new_stream
        current_ble_address = addr or "auto"
        onnx_session = _real_onnx_session
        return _ok({"mode": "ble", "ble_address": current_ble_address, "message": "BLE 已启动（自动扫描或指定地址）"})


# ── Sessions / 多端归档（后端统一规范与洞察）──────────────────────
@app.post("/api/v1/sessions/web/start", tags=["Sessions"], summary="网页开始记录")
async def v1_sessions_web_start():
    """仅设标志位；快照由 processing_loop 写入。前端无需传参。"""
    global web_recording
    async with _transport_lock:
        web_recording = {"started_at": time.time(), "_last_snap": 0.0, "snapshots": []}
    return _ok({"message": "已开始记录", "sample_interval_ms": 500})


@app.post("/api/v1/sessions/web/stop", tags=["Sessions"], summary="网页结束记录并归档")
async def v1_sessions_web_stop():
    global web_recording
    async with _transport_lock:
        rec = web_recording
        web_recording = None
    if not rec or not rec.get("snapshots"):
        return _err("no_data", "没有快照或未开始记录")
    pkg = _build_web_session_package(rec)
    canonicalize_export_package(pkg)
    insight = session_insight_text(pkg)
    avg_s = sum(s.get("stamina", 0) for s in pkg["snapshots"]) / len(pkg["snapshots"])
    pkg["summary"] = {
        "text": insight,
        "avgStamina": round(avg_s, 1),
        "minStamina": min(s.get("stamina", 0) for s in pkg["snapshots"]),
        "maxStamina": max(s.get("stamina", 0) for s in pkg["snapshots"]),
        "totalDurationSec": pkg["session"]["durationSec"],
        "workDurationSec": pkg["session"]["durationSec"],
        "restDurationSec": 0.0,
    }
    sid = save_session_package(pkg)
    return _ok(
        {
            "session_id": sid,
            "insight": insight,
            "snapshot_count": len(pkg["snapshots"]),
            "download_url": f"/api/v1/sessions/{sid}/file",
        }
    )


@app.post("/api/v1/sessions/ingest", tags=["Sessions"], summary="导入会话包（如手机导出 JSON）")
async def v1_sessions_ingest(body: Dict[str, Any] = Body(...)):
    """Body 与 iOS `ExportPackage` JSON 同构；后端规范化 8 路 RMS 并落盘。"""
    if not isinstance(body, dict) or "session" not in body:
        return _err("bad_request", "缺少 session 字段或 JSON 格式错误")
    canonicalize_export_package(body)
    insight = session_insight_text(body)
    snaps = body.get("snapshots") or []
    stamins = [float(s.get("stamina", 0)) for s in snaps if isinstance(s, dict)]
    sum_obj = body.get("summary")
    if not isinstance(sum_obj, dict):
        sum_obj = {}
        body["summary"] = sum_obj
    if not (sum_obj.get("text") or "").strip():
        sum_obj["text"] = insight
    sum_obj.setdefault("avgStamina", sum(stamins) / len(stamins) if stamins else 0)
    sum_obj.setdefault("minStamina", min(stamins) if stamins else 0)
    sum_obj.setdefault("maxStamina", max(stamins) if stamins else 0)
    dur = (body.get("session") or {}).get("durationSec", 0)
    sum_obj.setdefault("totalDurationSec", dur)
    sum_obj.setdefault("workDurationSec", dur)
    sum_obj.setdefault("restDurationSec", 0.0)
    sid = save_session_package(body)
    return _ok({"session_id": sid, "insight": insight})


@app.get("/api/v1/sessions", tags=["Sessions"], summary="已归档会话列表")
async def v1_sessions_list():
    return _ok({"sessions": list_sessions_meta()})


@app.get("/api/v1/sessions/{session_id}", tags=["Sessions"], summary="读取会话 JSON")
async def v1_sessions_get(session_id: str):
    try:
        return _ok(load_session(session_id))
    except ValueError:
        return _err("bad_request", "非法 session_id")
    except FileNotFoundError:
        return _err("not_found", "会话不存在")


@app.get("/api/v1/sessions/{session_id}/file", tags=["Sessions"], summary="下载会话 JSON 文件")
async def v1_sessions_download(session_id: str):
    try:
        path = session_file_path(session_id)
    except ValueError:
        return _err("bad_request", "非法 session_id")
    if not path.is_file():
        return _err("not_found", "会话不存在")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.get("/api/v1/sessions/{session_id}/insight", tags=["Sessions"], summary="仅返回洞察文案")
async def v1_sessions_insight(session_id: str):
    try:
        pkg = load_session(session_id)
    except ValueError:
        return _err("bad_request", "非法 session_id")
    except FileNotFoundError:
        return _err("not_found", "会话不存在")
    return _ok({"insight": session_insight_text(pkg)})


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


def _ensure_vision_stack() -> None:
    global vision_engine, fusion_engine, vision_module, context_bus
    if context_bus is None:
        context_bus = ContextBus()
    if vision_engine is None:
        vision_engine = VisionEngine()
    if vision_module is None:
        vision_module = VisionModule(engine=vision_engine, context_bus=context_bus)
    if fusion_engine is None:
        fusion_engine = FusionEngine(context_bus=context_bus)
        fusion_engine.register(vision_module)


@app.websocket("/ws/vision")
async def websocket_vision(ws: WebSocket):
    """浏览器 MediaPipe → vision_frame JSON；算法见 src/vision_engine.py。"""
    global _last_vision_frame_ts
    await ws.accept()
    _ensure_vision_stack()
    assert vision_engine is not None
    vision_ws_connections.add(ws)
    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "code": "bad_json", "message": "invalid JSON"})
                continue
            if data.get("type") == "ping":
                await ws.send_json({"type": "pong"})
                continue
            vision_engine.update(data)
            if data.get("type") == "vision_frame":
                _last_vision_frame_ts = time.time()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[vision ws] {e}")
    finally:
        vision_ws_connections.discard(ws)


def _append_vision_and_fusion(
    payload: dict,
    stamina_reading,
    emg_buffer: Optional[np.ndarray] = None,
) -> None:
    """写入 vision 快照与 EMG+Vision 融合结果。

    通过正式的 EmgModule / VisionModule 注入 ModuleReading，
    再调 FusionEngine.fuse() 完成 N-source 融合。
    """
    _ensure_vision_stack()
    now = time.time()

    # ── EMG 侧: 将 StaminaReading 写入 EmgModule ──
    if emg_module is not None and stamina_reading is not None:

        emg_q = (
            _emg_ac_quality_from_buffer(emg_buffer)
            if emg_buffer is not None and getattr(emg_buffer, "size", 0) > 0
            else EmgModule._estimate_quality(stamina_reading)
        )
        emg_module._set_latest(ModuleReading(
            module_id="emg",
            stamina=stamina_reading.stamina,
            quality=emg_q,
            dimensions={
                "consistency": stamina_reading.consistency,
                "tension": stamina_reading.tension,
                "fatigue_mdf": stamina_reading.fatigue,
            },
            metadata={"state": stamina_reading.state.value},
            alerts=[],
            timestamp=stamina_reading.timestamp,
        ))
    elif emg_module is not None:
        emg_module._latest = None

    # ── Vision 侧: 将 VisionReading 写入 VisionModule ──
    vr = vision_engine.latest if vision_engine else None
    vision_stale = (
        vr is not None
        and (
            _last_vision_frame_ts <= 0
            or (now - _last_vision_frame_ts) >= VISION_FRAME_STALE_SEC
        )
    )
    if vr:
        vd = vr.to_dict()
        vd["stale"] = vision_stale
        payload["vision"] = vd

    if vision_module is not None and vr is not None and not vision_stale:

        vision_module._set_latest(ModuleReading(
            module_id="vision",
            stamina=(1.0 - vr.fatigue_score) * 100.0,
            quality=vr.quality,
            dimensions={
                "perclos": vr.perclos,
                "blink_rate": vr.blink_rate,
                "fatigue_score": vr.fatigue_score,
            },
            metadata={
                "alertness_level": vr.alertness_level,
                "face_present": vr.face_present,
                "head_nod_detected": vr.head_nod_detected,
                "head_distracted": vr.head_distracted,
                "yawn_active": vr.yawn_active,
            },
            alerts=vision_reading_alerts(vr),
            timestamp=vr.timestamp,
        ))
    elif vision_module is not None:
        vision_module._latest = None

    # ── 融合 ──
    if fusion_engine:
        payload["fusion"] = fusion_engine.fuse(now).to_dict()


# ─── Processing loop ────────────────────────────────────────
async def processing_loop():
    global stamina_engine, decision_engine, _emg_buffer_reset_requested, emg_module

    stamina_engine = StaminaEngine(sample_rate=SAMPLE_RATE, speed=speed_multiplier)
    decision_engine = DecisionEngine()

    # 注册 EMG 模块到融合引擎
    _ensure_vision_stack()
    emg_module = EmgModule(engine=stamina_engine, context_bus=context_bus)
    if fusion_engine is not None:
        fusion_engine.register(emg_module)

    emg_buffer = np.zeros((0, CHANNEL_COUNT))
    emg_ts = np.array([], dtype=np.float64)
    last_push = time.time()
    total_received = 0
    log_interval = time.time()

    while True:
        if _emg_buffer_reset_requested:
            emg_buffer = np.zeros((0, CHANNEL_COUNT))
            emg_ts = np.array([], dtype=np.float64)
            total_received = 0
            _emg_buffer_reset_requested = False

        if stream is None:
            await asyncio.sleep(0.2)
            now = time.time()
            idle_payload = {
                "type": "state_update",
                "timestamp": now,
                "activity": "idle",
                "confidence": 0.0,
                "probabilities": {},
                "rms": [0.0] * CHANNEL_COUNT,
                "emg_sample_count": 0,
            }
            _append_vision_and_fusion(idle_payload, None, None)
            latest_payload.clear()
            latest_payload.update(idle_payload)
            await broadcast(idle_payload)
            continue

        samples = stream.consume_samples(max_items=512)
        if not samples:
            await asyncio.sleep(0.02)
            continue

        total_received += len(samples)
        new_data = np.array([s.values for s in samples])
        new_ts = np.array([s.timestamp for s in samples], dtype=np.float64)
        emg_buffer = np.vstack([emg_buffer, new_data]) if emg_buffer.size else new_data
        emg_ts = np.concatenate([emg_ts, new_ts]) if emg_ts.size else new_ts

        if emg_buffer.shape[0] > SAMPLE_RATE * 10:
            emg_buffer = emg_buffer[-SAMPLE_RATE * 5:]
            emg_ts = emg_ts[-SAMPLE_RATE * 5:]

        now = time.time()
        if now - log_interval >= 5.0:
            stats = stream.frame_stats()
            _log_seg = remove_dc(np.asarray(new_data, dtype=np.float64))
            rms_now = np.sqrt(np.mean(_log_seg**2, axis=0))
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

        eff_sr = _estimate_eff_sample_rate(emg_buffer.shape[0], emg_ts)
        window_samples = max(32, int(WINDOW_SEC * eff_sr))
        window_samples = min(window_samples, emg_buffer.shape[0])

        if onnx_session is not None and emg_buffer.shape[0] >= window_samples:
            window = emg_buffer[-window_samples:]
            try:
                feats = extract_window_features_84(window, int(round(eff_sr)))
                activity, confidence, probabilities = onnx_session.predict(feats)
            except Exception as e:
                print(f"[inference] {e}")

        stamina_reading = None
        if stamina_engine and emg_buffer.shape[0] >= window_samples:
            try:
                stamina_engine.set_effective_sample_rate(eff_sr)
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

        _rms_seg = emg_buffer[-min(250, len(emg_buffer)) :]
        _rms_seg = remove_dc(np.asarray(_rms_seg, dtype=np.float64))
        rms_values = np.sqrt(np.mean(_rms_seg**2, axis=0)).tolist()

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

        _append_vision_and_fusion(payload, stamina_reading, emg_buffer)

        latest_payload.clear()
        latest_payload.update(payload)

        _capture_web_recording_snapshot()

        await broadcast(payload)
        await asyncio.sleep(0.01)


# ─── Entrypoint ──────────────────────────────────────────────
def main():
    global stream, onnx_session, onnx_classes, onnx_config, demo_mode, speed_multiplier
    global _real_onnx_session, current_serial_port, current_ble_address

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
                loaded = ONNXClassifier(str(onnx_path), config)
                _real_onnx_session = loaded
                onnx_session = loaded
                onnx_config = config
                onnx_classes = config.get("classes", [])
                source = config.get("source", "unknown")
                print(f"[web] Loaded ONNX model: {onnx_path}")
                print(f"[web]   Classes: {onnx_classes}")
                print(f"[web]   Features: {config.get('n_features', '?')}")
                print(f"[web]   Accuracy: {config.get('accuracy', '?')} (on {source})")
                if "ninapro" in source.lower():
                    print(f"[web]   ** PLACEHOLDER: trained on {source}, NOT on your wristband **")
                    print(f"[web]   ** Run `python tools/emg_diagnostic.py` to verify signal quality **")
                break
            except Exception as e:
                print(f"[web] Failed to load {onnx_path}: {e}")

    if demo_mode:
        print("[web] DEMO mode (synthetic data)")
        stream = DemoEMGStream()
        stream.start()
        if _real_onnx_session is None:
            onnx_session = DemoClassifier(stream)
            print("[web] Using demo classifier")
        else:
            onnx_session = _real_onnx_session
    elif args.ble is not None:
        addr = None if args.ble == "auto" else args.ble
        print(f"[web] BLE mode — {'auto-scan' if addr is None else addr}")
        stream = BleEMGStream(address=addr)
        stream.start()
        current_ble_address = addr or "auto"
        onnx_session = _real_onnx_session
    elif args.port:
        print(f"[web] Connecting to {args.port}...")
        stream = SerialEMGStream(args.port)
        stream.start()
        current_serial_port = args.port
        onnx_session = _real_onnx_session
    else:
        print("[web] 未指定 --demo / --port / --ble — 数据源空闲，可在网页「数据源」中选择或调用 POST /api/v1/transport/apply")
        stream = None
        current_serial_port = None
        current_ble_address = None
        onnx_session = _real_onnx_session

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
    print(f"  Vision      http://localhost:{p}/static/vision.html  (调试页；主仪表盘已内置摄像头)")
    print(f"  Mode: {_transport_mode_label()}  Speed: {speed_multiplier}x\n")
    uvicorn.run(app, host=args.host, port=p, log_level="warning")


if __name__ == "__main__":
    main()
