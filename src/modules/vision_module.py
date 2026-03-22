"""
FluxChi -- Vision Module
=========================================================
包装 VisionEngine, 产出标准化 ModuleReading。

协作信号 (通过 ContextBus):
  - 发布 MICROSLEEP: 闭眼 > 500ms
  - 发布 PERCLOS_RISING: PERCLOS 持续上升
  - 发布 NOD_DETECTED: 低头打瞌睡
  - 发布 FACE_LOST: 人脸丢失

消费信号:
  - MUSCLE_TENSION_HIGH → 降低分心判定灵敏度
  - KEYBOARD_ACTIVE → 提高低头 pitch 容忍度
=========================================================
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..context_bus import ContextBus, ContextSignal, SignalTypes
from ..sensor_module import ModuleReading, SensorModule
from ..vision_engine import (
    PERCLOS_ALERT_SEVERE,
    VisionEngine,
    VisionReading,
    vision_reading_alerts,
)


class VisionModule(SensorModule):
    """摄像头视觉传感器模块。"""

    @property
    def module_id(self) -> str:
        return "vision"

    @property
    def display_name(self) -> str:
        return "摄像头视觉"

    def __init__(
        self,
        engine: Optional[VisionEngine] = None,
        context_bus: Optional[ContextBus] = None,
    ) -> None:
        super().__init__()
        self._engine = engine or VisionEngine()
        self._bus = context_bus
        self._prev_perclos: float = 0.0  # 上一次 PERCLOS, 用于趋势检测

    @property
    def engine(self) -> VisionEngine:
        """暴露底层引擎, 供 /ws/vision 端点直接调用。"""
        return self._engine

    @property
    def vision_reading(self) -> Optional[VisionReading]:
        """获取底层 VisionReading (向后兼容)。"""
        return self._engine.latest

    def update(self, frame: Dict[str, Any]) -> Optional[ModuleReading]:
        """
        处理一条 vision_frame, 转化为 ModuleReading。

        参数:
            frame: 符合 VISION-WEBSOCKET-SPEC.md 的 vision_frame dict

        返回:
            ModuleReading, 或 None (帧格式有误)
        """
        vr: Optional[VisionReading] = self._engine.update(frame)
        if vr is None:
            return None

        # Vision fatigue_score (0-1) → stamina (0-100)
        stamina = (1.0 - vr.fatigue_score) * 100.0

        alerts = self._collect_alerts(vr)

        reading = ModuleReading(
            module_id=self.module_id,
            stamina=stamina,
            quality=vr.quality,
            dimensions={
                "perclos": vr.perclos,
                "blink_rate": vr.blink_rate,
                "blink_count": float(vr.blink_count_window),
                "head_pitch_mean": vr.head_pitch_mean,
                "head_yaw_mean": vr.head_yaw_mean,
                "yawn_count": float(vr.yawn_count_window),
                "fatigue_score": vr.fatigue_score,
            },
            metadata={
                "alertness_level": vr.alertness_level,
                "face_present": vr.face_present,
                "head_nod_detected": vr.head_nod_detected,
                "head_distracted": vr.head_distracted,
                "yawn_active": vr.yawn_active,
                "frames_processed": vr.frames_processed,
            },
            alerts=alerts,
            timestamp=vr.timestamp,
        )

        self._set_latest(reading)
        self._publish_context(vr)
        self._prev_perclos = vr.perclos
        return reading

    # ── 告警收集 ──────────────────────────────────────────

    @staticmethod
    def _collect_alerts(vr: VisionReading) -> List[str]:
        """从 VisionReading 提取告警（实现为 ``vision_engine.vision_reading_alerts``）。"""
        return vision_reading_alerts(vr)

    # ── 协作信号 ──────────────────────────────────────────

    def _publish_context(self, vr: VisionReading) -> None:
        """向 ContextBus 发布协作信号。"""
        if self._bus is None:
            return

        now = vr.timestamp

        # 微睡眠 / 严重困倦
        if vr.perclos > PERCLOS_ALERT_SEVERE or vr.head_nod_detected:
            self._bus.publish(ContextSignal(
                source_module=self.module_id,
                signal_type=SignalTypes.MICROSLEEP,
                value={"perclos": vr.perclos, "nod": vr.head_nod_detected},
                timestamp=now,
                ttl_sec=30.0,
            ))

        # PERCLOS 上升趋势
        if vr.perclos > self._prev_perclos + 0.05 and vr.perclos > 0.15:
            self._bus.publish(ContextSignal(
                source_module=self.module_id,
                signal_type=SignalTypes.PERCLOS_RISING,
                value=vr.perclos,
                timestamp=now,
                ttl_sec=15.0,
            ))

        # 低头打瞌睡
        if vr.head_nod_detected:
            self._bus.publish(ContextSignal(
                source_module=self.module_id,
                signal_type=SignalTypes.NOD_DETECTED,
                value=True,
                timestamp=now,
                ttl_sec=20.0,
            ))

        # 人脸丢失
        if not vr.face_present:
            self._bus.publish(ContextSignal(
                source_module=self.module_id,
                signal_type=SignalTypes.FACE_LOST,
                value=True,
                timestamp=now,
                ttl_sec=5.0,
            ))

    def reset(self) -> None:
        self._engine.reset()
        self._latest = None
        self._prev_perclos = 0.0
