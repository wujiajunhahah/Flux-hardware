"""
FluxChi -- Context Bus (模块间协作层)
=========================================================
模块间传递上下文信号, 使各模块不只是独立打分,
而是互相知道彼此的状态并动态调整行为。

示例:
  - EMG 发布 "muscle_tension_high" → Vision 降低分心灵敏度
  - Vision 发布 "microsleep_detected" → Fusion 提高 Vision 权重
  - InputActivity 发布 "keyboard_active" → Vision 提高低头容忍度

设计:
  - 发布/查询模型 (非订阅回调), 保持简单可测试
  - 信号有 TTL, 过期自动清除
  - 线程安全 (asyncio 环境下单线程, 但预留 lock)
=========================================================
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ContextSignal:
    """模块间传递的上下文信号。"""

    source_module: str      # 发送方模块 ID
    signal_type: str        # 信号类型, 如 "muscle_tension_high"
    value: Any              # 信号值 (bool / float / dict)
    timestamp: float        # 发布时间
    ttl_sec: float = 10.0   # 生存时间 (秒), 超过后 query 不返回

    @property
    def expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl_sec


# ── 预定义信号类型 (约定, 非强制) ──────────────────────────

class SignalTypes:
    """信号类型常量, 方便模块之间约定。"""

    # Vision → 其他
    MICROSLEEP = "microsleep_detected"       # 微睡眠 (闭眼>500ms)
    PERCLOS_RISING = "perclos_rising"         # PERCLOS 持续上升
    NOD_DETECTED = "nod_detected"            # 低头打瞌睡
    FACE_LOST = "face_lost"                  # 人脸丢失

    # EMG → 其他
    MUSCLE_TENSION_HIGH = "muscle_tension_high"  # 肌肉紧张度高 (说明在工作)
    EMG_SIGNAL_LOST = "emg_signal_lost"          # EMG 信号丢失
    EMG_FATIGUE_MDF = "emg_fatigue_mdf"          # 中频频率下降 (肌肉疲劳)

    # InputActivity → 其他
    KEYBOARD_ACTIVE = "keyboard_active"      # 键盘活跃
    MOUSE_ACTIVE = "mouse_active"            # 鼠标活跃
    INPUT_IDLE = "input_idle"                # 无输入活动 (>N 分钟)

    # HeartRate → 其他
    HRV_DECLINING = "hrv_declining"          # 心率变异性下降
    HR_ELEVATED = "hr_elevated"              # 心率升高


class ContextBus:
    """模块间消息总线。发布/查询模型。

    用法:
        bus = ContextBus()

        # 模块 A 发布
        bus.publish(ContextSignal(
            source_module="emg",
            signal_type=SignalTypes.MUSCLE_TENSION_HIGH,
            value=True,
            timestamp=time.time(),
            ttl_sec=15.0,
        ))

        # 模块 B 查询
        signals = bus.query(SignalTypes.MUSCLE_TENSION_HIGH, max_age_sec=10.0)
        if signals:
            # 调整自身行为...
    """

    def __init__(self) -> None:
        self._signals: Dict[str, List[ContextSignal]] = defaultdict(list)
        self._lock = threading.Lock()

    def publish(self, signal: ContextSignal) -> None:
        """发布一个信号。"""
        with self._lock:
            bucket = self._signals[signal.signal_type]
            bucket.append(signal)
            # 清理同类型过期信号, 防止无限增长
            self._signals[signal.signal_type] = [
                s for s in bucket if not s.expired
            ]

    def query(
        self,
        signal_type: str,
        max_age_sec: Optional[float] = None,
        source_module: Optional[str] = None,
    ) -> List[ContextSignal]:
        """查询指定类型的有效信号。

        参数:
            signal_type: 要查询的信号类型
            max_age_sec: 最大年龄 (秒), None 则使用信号自身的 TTL
            source_module: 限定来源模块, None 则不限
        """
        now = time.time()
        with self._lock:
            results = []
            for s in self._signals.get(signal_type, []):
                if s.expired:
                    continue
                if max_age_sec is not None and (now - s.timestamp) > max_age_sec:
                    continue
                if source_module is not None and s.source_module != source_module:
                    continue
                results.append(s)
            return results

    def has_recent(
        self,
        signal_type: str,
        max_age_sec: float = 10.0,
        source_module: Optional[str] = None,
    ) -> bool:
        """快捷查询: 是否存在指定类型的近期信号。"""
        return len(self.query(signal_type, max_age_sec, source_module)) > 0

    def latest_value(
        self,
        signal_type: str,
        max_age_sec: float = 10.0,
        default: Any = None,
    ) -> Any:
        """快捷查询: 获取最近一条信号的 value。"""
        signals = self.query(signal_type, max_age_sec)
        if not signals:
            return default
        return max(signals, key=lambda s: s.timestamp).value

    def clear(self, signal_type: Optional[str] = None) -> None:
        """清除信号。signal_type=None 则全部清除。"""
        with self._lock:
            if signal_type is None:
                self._signals.clear()
            else:
                self._signals.pop(signal_type, None)

    def gc(self) -> int:
        """垃圾回收: 清除所有过期信号, 返回清除数量。"""
        removed = 0
        with self._lock:
            for key in list(self._signals.keys()):
                before = len(self._signals[key])
                self._signals[key] = [s for s in self._signals[key] if not s.expired]
                removed += before - len(self._signals[key])
                if not self._signals[key]:
                    del self._signals[key]
        return removed
