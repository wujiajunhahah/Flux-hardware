"""
FluxChi -- EMG Module
=========================================================
包装 StaminaEngine, 产出标准化 ModuleReading。
不修改 StaminaEngine 的任何逻辑, 只做适配。

协作信号 (通过 ContextBus):
  - 发布 MUSCLE_TENSION_HIGH: tension > 0.7
  - 发布 EMG_SIGNAL_LOST: 超时无数据
  - 发布 EMG_FATIGUE_MDF: fatigue (MDF) > 0.6
=========================================================
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..context_bus import ContextBus, ContextSignal, SignalTypes
from ..energy import StaminaEngine, StaminaReading
from ..sensor_module import ModuleReading, SensorModule


class EmgModule(SensorModule):
    """EMG 手环传感器模块。"""

    @property
    def module_id(self) -> str:
        return "emg"

    @property
    def display_name(self) -> str:
        return "EMG 手环"

    def __init__(
        self,
        engine: StaminaEngine,
        context_bus: Optional[ContextBus] = None,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._bus = context_bus

    @property
    def engine(self) -> StaminaEngine:
        """暴露底层引擎, 供 app.py 获取 StaminaReading 做向后兼容。"""
        return self._engine

    def update(
        self,
        emg_window: np.ndarray,
        activity: str,
        timestamp: Optional[float] = None,
    ) -> ModuleReading:
        """
        驱动 StaminaEngine.update(), 转化为 ModuleReading。

        参数:
            emg_window: (samples, channels) 的 EMG 数据窗口
            activity: 当前检测到的活动类型
            timestamp: 可选时间戳
        """
        sr: StaminaReading = self._engine.update(emg_window, activity, timestamp)

        reading = ModuleReading(
            module_id=self.module_id,
            stamina=sr.stamina,
            quality=self._estimate_quality(sr),
            dimensions={
                "consistency": sr.consistency,
                "tension": sr.tension,
                "fatigue_mdf": sr.fatigue,
            },
            metadata={
                "state": sr.state.value,
                "drain_rate": sr.drain_rate,
                "recovery_rate": sr.recovery_rate,
                "suggested_work_min": sr.suggested_work_min,
                "suggested_break_min": sr.suggested_break_min,
                "continuous_work_min": sr.continuous_work_min,
                "total_work_min": sr.total_work_min,
            },
            alerts=[],
            timestamp=sr.timestamp,
        )

        self._set_latest(reading)
        self._publish_context(sr)
        return reading

    @property
    def stamina_reading(self) -> Optional[StaminaReading]:
        """获取底层 StaminaReading (向后兼容 app.py 中构建 payload 的逻辑)。"""
        return self._engine.latest if hasattr(self._engine, "latest") else None

    # ── 内部 ──────────────────────────────────────────────

    @staticmethod
    def _estimate_quality(sr: StaminaReading) -> float:
        """EMG 信号质量估算 (无原始波形时用工况维度代理)。"""
        c = float(np.clip(sr.consistency, 0.0, 1.0))
        t = float(np.clip(sr.tension, 0.0, 1.0))
        return float(np.clip(0.2 + 0.65 * c + 0.15 * (1.0 - t), 0.2, 1.0))

    def _publish_context(self, sr: StaminaReading) -> None:
        """向 ContextBus 发布协作信号。"""
        if self._bus is None:
            return

        now = sr.timestamp

        # 肌肉紧张度高 → 告知 Vision 用户在工作, 降低分心灵敏度
        if sr.tension > 0.7:
            self._bus.publish(ContextSignal(
                source_module=self.module_id,
                signal_type=SignalTypes.MUSCLE_TENSION_HIGH,
                value=sr.tension,
                timestamp=now,
                ttl_sec=15.0,
            ))

        # MDF 疲劳指标高 → 肌肉疲劳先兆
        if sr.fatigue > 0.6:
            self._bus.publish(ContextSignal(
                source_module=self.module_id,
                signal_type=SignalTypes.EMG_FATIGUE_MDF,
                value=sr.fatigue,
                timestamp=now,
                ttl_sec=15.0,
            ))

    def reset(self) -> None:
        self._engine.reset()
        self._latest = None
