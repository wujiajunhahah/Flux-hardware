"""
FluxChi -- Sensor Module 标准接口
=========================================================
所有传感器模块 (EMG 手环 / 摄像头 / 键鼠 / 心率 / ...)
遵循同一接口, 产出 ModuleReading, 供 FusionEngine 统一消费。

设计原则:
  - 统一量纲: stamina 0-100, quality 0-1
  - 模块自治: 每个模块独立维护状态, 独立判断质量
  - 协作通道: 通过 ContextBus 发布/订阅跨模块信号
=========================================================
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  标准化读数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ModuleReading:
    """所有传感器模块的标准化输出。

    统一量纲:
      - stamina: 0-100, 越高越精力充沛 (EMG 原生; Vision 由 1-fatigue_score 映射)
      - quality: 0-1,   信号质量, 供 FusionEngine 降权/忽略
    """

    module_id: str                  # 唯一标识: "emg" / "vision" / "input" / ...
    stamina: float                  # 0-100
    quality: float                  # 0-1

    # 模块特有维度 (名称由各模块自定义)
    # EMG:    {"consistency": 0.8, "tension": 0.3, "fatigue_mdf": 0.4}
    # Vision: {"perclos": 0.12, "blink_rate": 18.0, "head_pitch_mean": 8.5}
    dimensions: Dict[str, float] = field(default_factory=dict)

    # 任意元数据 (不参与融合计算, 用于前端展示/日志)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 模块级告警
    alerts: List[str] = field(default_factory=list)

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "stamina": round(self.stamina, 1),
            "quality": round(self.quality, 2),
            "dimensions": {k: round(v, 3) if isinstance(v, float) else v
                           for k, v in self.dimensions.items()},
            "metadata": self.metadata,
            "alerts": self.alerts,
            "timestamp": self.timestamp,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  传感器模块抽象基类
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SensorModule(ABC):
    """传感器模块基类。

    子类需实现:
      - module_id (属性): 唯一标识
      - display_name (属性): 人类可读名
      - reset(): 重置内部状态

    约定:
      - update() 方法签名由子类自定义 (各模块输入格式不同)
      - update() 内部调用 self._set_latest(reading) 保存结果
    """

    def __init__(self) -> None:
        self._latest: Optional[ModuleReading] = None

    @property
    @abstractmethod
    def module_id(self) -> str:
        """唯一标识, 如 'emg', 'vision', 'input_activity'。"""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """人类可读名, 如 'EMG 手环', '摄像头视觉'。"""
        ...

    @property
    def latest(self) -> Optional[ModuleReading]:
        """最近一次 ModuleReading。"""
        return self._latest

    @property
    def is_available(self) -> bool:
        """模块是否有过有效读数。子类可覆写以加入连接状态判断。"""
        return self._latest is not None

    def stale_seconds(self, now: Optional[float] = None) -> float:
        """距离最近读数的秒数。无读数时返回 inf。"""
        if self._latest is None:
            return float("inf")
        return (now or time.time()) - self._latest.timestamp

    def _set_latest(self, reading: ModuleReading) -> None:
        """子类的 update() 内调用此方法保存读数。"""
        self._latest = reading

    @abstractmethod
    def reset(self) -> None:
        """重置模块内部状态 (校准/重连时调用)。"""
        ...
