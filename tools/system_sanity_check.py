#!/usr/bin/env python3
"""
FluxChi 离线系统自检（无串口 / 无摄像头 / 不启动 HTTP）

用途：
  - 回归：VisionEngine / FusionEngine / StaminaEngine / EMG 模块管线是否可导入并输出合理结构
  - 调试：合成 vision_frame 与 EMG 窗口，打印断言失败原因

与设计文档对齐：模块自治 + 融合层质量门控，见 docs/ARCHITECTURE-V2.md。

用法：
  python tools/system_sanity_check.py
  python tools/system_sanity_check.py --verbose
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context_bus import ContextBus
from src.energy import StaminaEngine
from src.features import remove_dc
from src.fusion_engine import FusionConfig, FusionEngine
from src.modules.emg_module import EmgModule
from src.modules.vision_module import VisionModule
from src.vision_engine import (
    PERCLOS_ALERT_HIGH,
    VisionConfig,
    VisionEngine,
    VisionReading,
    vision_reading_alerts,
)


@dataclass
class CaseResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class RunReport:
    results: List[CaseResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append(CaseResult(name, ok, detail))
        sym = "OK " if ok else "FAIL"
        line = f"  [{sym}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [WARN] {msg}")


def _vf(
    t: float,
    seq: int,
    *,
    present: bool,
    geometry: Optional[Dict[str, Any]] = None,
    blendshapes: Optional[Dict[str, float]] = None,
    schema: int = 2,
) -> Dict[str, Any]:
    frame: Dict[str, Any] = {
        "type": "vision_frame",
        "schema": schema,
        "t": t,
        "seq": seq,
        "face": {"present": present, "confidence": 0.92 if present else 0.0},
    }
    if geometry is not None:
        frame["geometry"] = geometry
    if blendshapes is not None:
        frame["blendshapes"] = blendshapes
    return frame


def _geo_open() -> Dict[str, Any]:
    return {
        "earLeft": 0.32,
        "earRight": 0.30,
        "ear": 0.30,
        "mar": 0.15,
        "headPoseDeg": {"pitch": 5.0, "yaw": 4.0, "roll": 1.0},
    }


def _geo_closed_eyes() -> Dict[str, Any]:
    g = _geo_open()
    g["earLeft"] = g["earRight"] = g["ear"] = 0.08
    return g


def check_vision_engine_invalid_frames(r: RunReport) -> None:
    cfg = VisionConfig(ear_blink_consec_frames=1)
    eng = VisionEngine(config=cfg)
    r.add("vision rejects wrong type", eng.update({"type": "ping"}) is None)
    r.add("vision rejects missing t", eng.update({"type": "vision_frame", "seq": 1}) is None)


def check_vision_no_face(r: RunReport) -> None:
    cfg = VisionConfig()
    eng = VisionEngine(config=cfg)
    vr = eng.update(_vf(100.0, 1, present=False))
    ok = vr is not None and vr.face_present is False and vr.perclos == 0.0
    r.add("vision no-face snapshot", ok, f"perclos={getattr(vr, 'perclos', None)}")


def check_vision_blink_geometry(r: RunReport) -> None:
    """短窗口 + 单帧即判闭眼，便于单元式验证眨眼状态机。"""
    cfg = VisionConfig(
        ear_blink_threshold=0.18,
        ear_blink_consec_frames=1,
        blink_rate_window_sec=3.0,
        perclos_window_sec=3.0,
    )
    eng = VisionEngine(config=cfg)
    t0 = 200.0
    n = 0
    for i in range(40):
        eng.update(_vf(t0 + i * 0.03, n := n + 1, present=True, geometry=_geo_open()))
    for i in range(4):
        eng.update(_vf(t0 + 40 * 0.03 + i * 0.03, n := n + 1, present=True, geometry=_geo_closed_eyes()))
    for i in range(10):
        eng.update(_vf(t0 + 44 * 0.03 + i * 0.03, n := n + 1, present=True, geometry=_geo_open()))
    vr = eng.latest
    assert vr is not None
    ok = vr.blink_count_window >= 1 and vr.blink_rate > 0
    r.add(
        "vision blink (geometry EAR)",
        ok,
        f"blink_count={vr.blink_count_window}, rate={vr.blink_rate:.2f}/min",
    )


def check_vision_blink_blendshapes(r: RunReport) -> None:
    cfg = VisionConfig(
        bs_blink_threshold=0.5,
        bs_blink_consec_frames=1,
        blink_rate_window_sec=2.0,
        perclos_window_sec=2.0,
        use_blendshapes_for_blink=True,
    )
    eng = VisionEngine(config=cfg)
    t0 = 300.0
    n = 0
    open_bs = {"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05}
    shut_bs = {"eyeBlinkLeft": 0.85, "eyeBlinkRight": 0.88}
    for i in range(25):
        eng.update(
            _vf(t0 + i * 0.04, n := n + 1, present=True, geometry=_geo_open(), blendshapes=open_bs)
        )
    for i in range(3):
        eng.update(
            _vf(t0 + 25 * 0.04 + i * 0.04, n := n + 1, present=True, geometry=_geo_open(), blendshapes=shut_bs)
        )
    for i in range(8):
        eng.update(
            _vf(t0 + 28 * 0.04 + i * 0.04, n := n + 1, present=True, geometry=_geo_open(), blendshapes=open_bs)
        )
    vr = eng.latest
    assert vr is not None
    ok = vr.blink_count_window >= 1
    r.add(
        "vision blink (blendshapes)",
        ok,
        f"blink_count={vr.blink_count_window}",
    )


def check_vision_single_eye_blink(r: RunReport) -> None:
    cfg = VisionConfig(
        bs_blink_threshold=0.55,
        bs_blink_consec_frames=1,
        blink_rate_window_sec=2.0,
        perclos_window_sec=2.0,
    )
    eng = VisionEngine(config=cfg)
    t0 = 400.0
    n = 0
    for i in range(15):
        eng.update(_vf(t0 + i * 0.05, n := n + 1, present=True, blendshapes={"eyeBlinkLeft": 0.1}))
    for i in range(2):
        eng.update(_vf(t0 + 15 * 0.05 + i * 0.05, n := n + 1, present=True, blendshapes={"eyeBlinkLeft": 0.75}))
    for i in range(6):
        eng.update(_vf(t0 + 17 * 0.05 + i * 0.05, n := n + 1, present=True, blendshapes={"eyeBlinkLeft": 0.1}))
    vr = eng.latest
    assert vr is not None
    ok = vr.blink_count_window >= 1
    r.add("vision blink (single eye BS)", ok, f"blink_count={vr.blink_count_window}")


def check_vision_perclos_alert_constants(r: RunReport) -> None:
    r.add("PERCLOS_ALERT_HIGH is 0.25", PERCLOS_ALERT_HIGH == 0.25, str(PERCLOS_ALERT_HIGH))


def check_vision_reading_alerts_fn(r: RunReport) -> None:
    vr = VisionReading(
        perclos=0.5,
        blink_rate=30.0,
        blink_count_window=0,
        head_pitch_mean=0.0,
        head_yaw_mean=0.0,
        head_nod_detected=True,
        head_distracted=True,
        yawn_count_window=4,
        yawn_active=True,
        fatigue_score=0.7,
        alertness_level="severe_fatigue",
        quality=0.85,
        face_present=True,
        frames_processed=20,
    )
    a = vision_reading_alerts(vr)
    ok = (
        a[0] == "nod_detected"
        and "perclos_severe" in a
        and "perclos_high" not in a
        and "yawn_frequent" in a
        and "blink_rate_high" in a
        and "distracted" in a
        and "yawning" in a
    )
    r.add("vision_reading_alerts() ordering", ok, str(a))


def check_stamina_engine(r: RunReport) -> None:
    eng = StaminaEngine(sample_rate=400, speed=1.0)
    rng = np.random.default_rng(42)
    w = rng.normal(0, 80.0, size=(120, 8))
    try:
        sr = eng.update(w, "typing", timestamp=500.0)
        ok = 0 <= sr.stamina <= 100 and sr.state is not None
        r.add(
            "stamina engine update",
            ok,
            f"stamina={sr.stamina:.1f} state={sr.state.value}",
        )
    except Exception as e:
        r.add("stamina engine update", False, str(e))


def check_remove_dc(r: RunReport) -> None:
    x = np.ones((100, 8)) * 500.0 + np.random.default_rng(0).normal(0, 10, size=(100, 8))
    y = remove_dc(x)
    mean_abs = float(np.mean(np.abs(np.mean(y, axis=0))))
    ok = mean_abs < 1.0
    r.add("remove_dc zero-mean columns", ok, f"mean_abs={mean_abs:.4f}")


def check_fusion_emg_vision(r: RunReport) -> None:
    bus = ContextBus()
    fe = FusionEngine(config=FusionConfig(), context_bus=bus)
    vcfg = VisionConfig(blink_rate_window_sec=5.0, perclos_window_sec=5.0)
    ve = VisionEngine(config=vcfg)
    vm = VisionModule(engine=ve, context_bus=bus)
    se = StaminaEngine(sample_rate=300, speed=1.0)
    em = EmgModule(engine=se, context_bus=bus)
    fe.register(em)
    fe.register(vm)

    t0 = 600.0
    n = 0
    for i in range(30):
        vm.update(_vf(t0 + i * 0.05, n := n + 1, present=True, geometry=_geo_open()))

    rng = np.random.default_rng(7)
    em.update(rng.normal(0, 60.0, size=(90, 8)), "typing", timestamp=t0 + 1.5)

    fused = fe.fuse(now=t0 + 2.0)
    ok = (
        fused.source in ("fused", "emg_only", "vision_only", "multi_fused", "none")
        and fused.stamina is not None
    )
    r.add(
        "fusion emg+vision",
        ok,
        f"source={fused.source} stamina={fused.stamina}",
    )


def check_fusion_vision_quality_gate(r: RunReport) -> None:
    """vision quality 过低时应被剔除，仅剩 EMG 或空。"""
    bus = ContextBus()
    # 正常 5 帧人脸时 quality 约 0.97+，取 0.98 确保 vision 被门限挡掉
    fe = FusionEngine(config=FusionConfig(quality_gate_vision=0.98), context_bus=bus)
    ve = VisionEngine()
    vm = VisionModule(engine=ve, context_bus=bus)
    se = StaminaEngine(sample_rate=300, speed=1.0)
    em = EmgModule(engine=se, context_bus=bus)
    fe.register(em)
    fe.register(vm)

    t0 = 700.0
    n = 0
    for i in range(5):
        vm.update(_vf(t0 + i * 0.1, n := n + 1, present=True, geometry=_geo_open()))
    rng = np.random.default_rng(8)
    em.update(rng.normal(0, 70.0, size=(100, 8)), "typing", timestamp=t0 + 0.5)

    fused = fe.fuse(now=t0 + 1.0)
    vw = fused.module_weights.get("vision", 0.0)
    ok = vw < 0.01 and fused.source == "emg_only"
    r.add(
        "fusion vision quality gate (strict)",
        ok,
        f"vision_weight={vw:.3f} source={fused.source}",
    )


def check_onnx_optional(r: RunReport, verbose: bool) -> None:
    model_dir = ROOT / "model"
    onnx_names = ("activity_classifier.onnx", "ninapro_classifier.onnx")
    path = next((model_dir / n for n in onnx_names if (model_dir / n).exists()), None)
    if path is None:
        r.warn("未找到 model/*.onnx，跳过 ONNX 推理检查")
        return
    cfg_path = path.with_name(path.stem + "_config.json")
    if not cfg_path.is_file():
        r.warn(f"缺少 {cfg_path.name}，跳过 ONNX")
        return
    try:
        import json

        import onnxruntime as ort
    except ImportError:
        r.warn("未安装 onnxruntime，跳过 ONNX")
        return

    config = json.loads(cfg_path.read_text())
    sess = ort.InferenceSession(str(path))
    in_name = sess.get_inputs()[0].name
    n_feat = int(config.get("n_features", 84))
    x = np.random.randn(1, n_feat).astype(np.float32)
    out = sess.run(None, {in_name: x})
    ok = len(out) >= 1 and np.asarray(out[0]).size >= 1
    r.add("ONNX session run (smoke)", ok, f"model={path.name}")
    if verbose:
        r.warn(f"ONNX 输出[0] shape={np.asarray(out[0]).shape}")


def run_all(verbose: bool) -> int:
    print("FluxChi system_sanity_check — 离线合成用例\n")
    rep = RunReport()

    sections: List[tuple[str, List[Callable[[RunReport], None]]]] = [
        ("特征 / EMG 基础", [check_remove_dc, check_stamina_engine]),
        ("VisionEngine", [
            check_vision_engine_invalid_frames,
            check_vision_no_face,
            check_vision_perclos_alert_constants,
            check_vision_reading_alerts_fn,
            check_vision_blink_geometry,
            check_vision_blink_blendshapes,
            check_vision_single_eye_blink,
        ]),
        ("FusionEngine", [check_fusion_emg_vision, check_fusion_vision_quality_gate]),
    ]

    for title, funcs in sections:
        print(f"── {title} ──")
        for fn in funcs:
            try:
                fn(rep)
            except Exception as e:
                rep.add(fn.__name__, False, f"exception: {e}")
        print()

    print("── 可选 ──")
    check_onnx_optional(rep, verbose)
    print()

    failed = sum(1 for x in rep.results if not x.ok)
    total = len(rep.results)
    print(f"合计: {total - failed}/{total} 通过", end="")
    if failed:
        print(f"  ({failed} 失败)")
    else:
        print()
    if rep.warnings and verbose:
        print("\nWarnings:")
        for w in rep.warnings:
            print(f"  - {w}")
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="FluxChi 离线系统自检")
    ap.add_argument("--verbose", "-v", action="store_true", help="更多提示")
    args = ap.parse_args()
    sys.exit(run_all(args.verbose))


if __name__ == "__main__":
    main()
