"""
FluxChi -- OpenFace 3.0 adapter
=========================================================
后端 OpenFace 接入层：
  1) 优先消费前端/外部进程注入的 per-frame openface 字段
  2) 缺失时根据几何/BlendShapes 做轻量兜底，确保 payload 稳定

说明：
  - 不强耦合具体推理后端，便于后续替换为真实 OpenFace 推理服务
  - 输出统一为 au / gaze / emotion，供 /ws 与 flywheel 训练使用
=========================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class OpenFaceFrame:
    au: Dict[str, float]
    gaze: Dict[str, float]
    emotion: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "au": self.au,
            "gaze": self.gaze,
            "emotion": self.emotion,
        }


class OpenFaceAdapter:
    """OpenFace 每帧特征适配器。"""

    def __init__(self) -> None:
        self._latest: Optional[OpenFaceFrame] = None

    @property
    def latest(self) -> Optional[OpenFaceFrame]:
        return self._latest

    def update(self, frame: Dict[str, Any]) -> Optional[OpenFaceFrame]:
        if frame.get("type") != "vision_frame":
            return self._latest

        direct = frame.get("openface")
        if isinstance(direct, dict):
            out = OpenFaceFrame(
                au=self._to_float_map(direct.get("au")),
                gaze=self._to_float_map(direct.get("gaze")),
                emotion=self._to_float_map(direct.get("emotion")),
            )
            self._latest = out
            return out

        # 兜底：从 geometry/blendshapes 近似构造，保证字段持续可用
        geo = frame.get("geometry") or {}
        bs = frame.get("blendshapes") or {}
        pose = geo.get("headPoseDeg") or {}

        eye_blink = max(
            float(bs.get("eyeBlinkLeft", 0.0) or 0.0),
            float(bs.get("eyeBlinkRight", 0.0) or 0.0),
        )
        jaw_open = float(bs.get("jawOpen", 0.0) or 0.0)
        brow_down = max(
            float(bs.get("browDownLeft", 0.0) or 0.0),
            float(bs.get("browDownRight", 0.0) or 0.0),
        )
        yaw = float(pose.get("yaw", 0.0) or 0.0)
        pitch = float(pose.get("pitch", 0.0) or 0.0)

        au = {
            "AU45_blink": min(1.0, eye_blink),
            "AU26_jawDrop": min(1.0, jaw_open),
            "AU04_browLowerer": min(1.0, brow_down),
        }
        gaze = {
            "x": max(-1.0, min(1.0, yaw / 40.0)),
            "y": max(-1.0, min(1.0, -pitch / 30.0)),
        }
        fatigue_bias = min(1.0, 0.7 * eye_blink + 0.3 * jaw_open)
        emotion = {
            "neutral": max(0.0, 1.0 - fatigue_bias),
            "fatigue": fatigue_bias,
        }
        out = OpenFaceFrame(au=au, gaze=gaze, emotion=emotion)
        self._latest = out
        return out

    @staticmethod
    def _to_float_map(value: Any) -> Dict[str, float]:
        if not isinstance(value, dict):
            return {}
        out: Dict[str, float] = {}
        for k, v in value.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out

