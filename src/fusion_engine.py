"""
FluxChi -- Fusion Engine V2
=========================================================
N-source 质量加权融合 + 层次化决策 + ContextBus 协作。

架构:
  1. 所有传感器通过 SensorModule → ModuleReading 统一接口
  2. FusionEngine.register() 注册模块, fuse() 自动从 latest 融合
  3. ContextBus 信号影响动态权重 (如 Vision 检测到微睡眠 → 权重↑)
  4. 层次化告警覆盖线性加权 (Level 3 事件直接降分)

向后兼容:
  - FusedReading.to_dict() 保留 emg_weight, vision_weight 等旧字段
  - fuse_legacy() 兼容旧的 fuse(emg=, vision=) 调用方式
  - payload["fusion"]["source"] 保持 "fused"/"emg_only"/"vision_only"/"none"
=========================================================
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .context_bus import ContextBus, SignalTypes
from .energy import StaminaReading
from .sensor_module import ModuleReading, SensorModule
from .vision_engine import VisionReading


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class FusionConfig:
    # 质量门控
    quality_gate: float = 0.3           # quality < 此值的模块被排除
    stale_timeout_sec: float = 5.0      # 超过此时间无更新的模块被排除

    # 模块基础权重 (module_id → weight), 未列出的默认 1.0
    base_weights: Dict[str, float] = field(default_factory=lambda: {
        "emg": 0.65,
        "vision": 0.35,
    })

    # 告警惩罚规则 (alert_name → stamina 扣分)
    alert_penalties: Dict[str, float] = field(default_factory=lambda: {
        "nod_detected": 10.0,
        "yawn_frequent": 5.0,
        "perclos_severe": 15.0,
    })

    # ── 向后兼容 (过渡期后删除) ──
    vision_quality_gate: float = 0.3
    emg_weight: float = 0.65
    vision_weight: float = 0.35
    yawn_count_alert: int = 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  输出
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class FusedReading:
    """融合后的综合判断。"""

    # 核心
    stamina: Optional[float]            # 0-100, None = 无有效数据
    state: str                          # focused / fading / depleted / unknown

    # N-source 权重明细
    module_weights: Dict[str, float]    # {module_id: actual_weight}
    source: str                         # "none" / "emg_only" / "vision_only" / "fused" / "multi_fused"

    # 各模块原始快照
    module_readings: Dict[str, Dict[str, Any]]  # {module_id: reading.to_dict()}

    # 合并告警
    alerts: List[str]

    # ── 向后兼容字段 ──
    emg_weight_actual: float = 0.0
    vision_weight_actual: float = 0.0
    emg_stamina: Optional[float] = None
    vision_fatigue: Optional[float] = None
    vision_quality: Optional[float] = None
    vision_alertness: Optional[str] = None

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            # 核心
            "stamina": round(self.stamina, 1) if self.stamina is not None else None,
            "state": self.state,
            "source": self.source,

            # N-source (新)
            "module_weights": {k: round(v, 3) for k, v in self.module_weights.items()},
            "modules": self.module_readings,

            # 合并告警
            "alerts": self.alerts,

            # 向后兼容
            "emg_weight": round(self.emg_weight_actual, 2),
            "vision_weight": round(self.vision_weight_actual, 2),
            "emg_stamina": round(self.emg_stamina, 1) if self.emg_stamina is not None else None,
            "vision_fatigue": round(self.vision_fatigue, 3) if self.vision_fatigue is not None else None,
            "vision_quality": round(self.vision_quality, 2) if self.vision_quality is not None else None,
            "vision_alertness": self.vision_alertness,

            "timestamp": self.timestamp,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  融合引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FusionEngine:
    """
    N-source 质量加权融合引擎。

    算法:
      1. 收集所有注册模块的 latest reading
      2. 过滤: quality < gate 或 stale 的模块被排除
      3. ContextBus 动态权重调制
      4. 归一化 + 加权平均 stamina
      5. 层次化告警覆盖 (Level 3 事件直接降分)
      6. 输出 FusedReading
    """

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        context_bus: Optional[ContextBus] = None,
    ) -> None:
        self.config = config or FusionConfig()
        self.latest: Optional[FusedReading] = None
        self._modules: Dict[str, SensorModule] = {}
        self._context_bus = context_bus or ContextBus()

    @property
    def context_bus(self) -> ContextBus:
        return self._context_bus

    @property
    def modules(self) -> Dict[str, SensorModule]:
        return dict(self._modules)

    def register(self, module: SensorModule) -> None:
        """注册传感器模块。"""
        self._modules[module.module_id] = module

    def unregister(self, module_id: str) -> None:
        """注销传感器模块。"""
        self._modules.pop(module_id, None)

    # ── 核心融合 ──────────────────────────────────────────

    def fuse(self, now: Optional[float] = None) -> FusedReading:
        """从所有注册模块的 latest 读数融合。"""
        now = now or time.time()
        cfg = self.config

        # Step 1: 收集有效读数
        active: List[ModuleReading] = []
        for module in self._modules.values():
            r = module.latest
            if r is None:
                continue
            if r.quality < cfg.quality_gate:
                continue
            if module.stale_seconds(now) > cfg.stale_timeout_sec:
                continue
            active.append(r)

        if not active:
            return self._build_empty(now)

        # Step 2: 计算权重 (base_weight * quality_factor * context_boost)
        raw_weights: Dict[str, float] = {}
        for r in active:
            base_w = cfg.base_weights.get(r.module_id, 1.0)
            q_factor = self._quality_to_factor(r.quality)
            c_boost = self._context_weight_boost(r.module_id, now)
            raw_weights[r.module_id] = base_w * q_factor * c_boost

        total_w = sum(raw_weights.values())
        if total_w < 1e-9:
            return self._build_empty(now)
        norm_weights = {k: v / total_w for k, v in raw_weights.items()}

        # Step 3: 加权平均 stamina
        fused_stamina = sum(
            norm_weights[r.module_id] * r.stamina for r in active
        )

        # Step 4: 收集告警 + 层次化覆盖
        all_alerts: List[str] = []
        for r in active:
            all_alerts.extend(r.alerts)
        all_alerts = list(dict.fromkeys(all_alerts))  # 去重保序

        # 告警惩罚
        penalty = sum(cfg.alert_penalties.get(a, 0.0) for a in all_alerts)
        fused_stamina = max(0.0, min(100.0, fused_stamina - penalty))

        # 层次化覆盖: Level 3 事件 (微睡眠/持续点头/PERCLOS 极高) → 强制低分
        level3 = {"nod_detected", "perclos_severe", "microsleep_detected"}
        if level3 & set(all_alerts):
            fused_stamina = min(fused_stamina, 15.0)

        state = _stamina_to_state(fused_stamina)
        source = self._determine_source(norm_weights)

        # Step 5: 构建 FusedReading (含向后兼容字段)
        module_readings = {r.module_id: r.to_dict() for r in active}

        reading = FusedReading(
            stamina=fused_stamina,
            state=state,
            module_weights=norm_weights,
            source=source,
            module_readings=module_readings,
            alerts=all_alerts,
            # 向后兼容
            emg_weight_actual=norm_weights.get("emg", 0.0),
            vision_weight_actual=norm_weights.get("vision", 0.0),
            emg_stamina=self._find_stamina("emg", active),
            vision_fatigue=self._find_dimension("vision", active, "fatigue_score"),
            vision_quality=self._find_field("vision", active, "quality"),
            vision_alertness=self._find_metadata("vision", active, "alertness_level"),
            timestamp=now,
        )
        self.latest = reading
        return reading

    # ── 向后兼容入口 (过渡期) ─────────────────────────────

    def fuse_legacy(
        self,
        emg: Optional[StaminaReading] = None,
        vision: Optional[VisionReading] = None,
    ) -> FusedReading:
        """兼容旧的 fuse(emg=, vision=) 调用。

        内部: 手动构建 ModuleReading 注入对应模块的 _latest, 再调 fuse()。
        过渡完成后删除此方法。
        """
        now = time.time()

        # 注入 EMG
        emg_mod = self._modules.get("emg")
        if emg is not None and emg_mod is not None:
            emg_mod._latest = ModuleReading(
                module_id="emg",
                stamina=emg.stamina,
                quality=1.0,
                dimensions={
                    "consistency": emg.consistency,
                    "tension": emg.tension,
                    "fatigue_mdf": emg.fatigue,
                },
                metadata={"state": emg.state.value},
                alerts=[],
                timestamp=emg.timestamp,
            )
        elif emg_mod is not None:
            emg_mod._latest = None

        # 注入 Vision
        vis_mod = self._modules.get("vision")
        if vision is not None and vis_mod is not None:
            vis_mod._latest = ModuleReading(
                module_id="vision",
                stamina=(1.0 - vision.fatigue_score) * 100.0,
                quality=vision.quality,
                dimensions={
                    "perclos": vision.perclos,
                    "blink_rate": vision.blink_rate,
                    "fatigue_score": vision.fatigue_score,
                },
                metadata={
                    "alertness_level": vision.alertness_level,
                    "face_present": vision.face_present,
                    "head_nod_detected": vision.head_nod_detected,
                    "head_distracted": vision.head_distracted,
                    "yawn_active": vision.yawn_active,
                },
                alerts=self._collect_vision_alerts(vision),
                timestamp=vision.timestamp,
            )
        elif vis_mod is not None:
            vis_mod._latest = None

        return self.fuse(now)

    @staticmethod
    def _collect_vision_alerts(vision: VisionReading) -> List[str]:
        """从 VisionReading 提取告警 (向后兼容, 旧路径用)。"""
        alerts: List[str] = []
        if vision.head_nod_detected:
            alerts.append("nod_detected")
        if vision.head_distracted:
            alerts.append("distracted")
        if vision.yawn_active:
            alerts.append("yawning")
        if vision.yawn_count_window >= 3:
            alerts.append("yawn_frequent")
        if vision.perclos > 0.40:
            alerts.append("perclos_severe")
        elif vision.perclos > 0.25:
            alerts.append("perclos_high")
        if vision.blink_rate > 25:
            alerts.append("blink_rate_high")
        return alerts

    # ── ContextBus 动态权重调制 ───────────────────────────

    def _context_weight_boost(self, module_id: str, now: float) -> float:
        """基于 ContextBus 信号动态调整模块权重。

        场景:
          - Vision 检测到微睡眠 → Vision 权重 x1.5 (更可信)
          - EMG 检测到肌肉紧张 → EMG 权重 x1.2 (在工作, EMG 更可信)
          - Vision 人脸丢失 → Vision 权重 x0.5
        """
        bus = self._context_bus
        boost = 1.0

        if module_id == "vision":
            # 自己检测到严重事件 → 提升自身权重
            if bus.has_recent(SignalTypes.MICROSLEEP, max_age_sec=30.0):
                boost *= 1.5
            if bus.has_recent(SignalTypes.PERCLOS_RISING, max_age_sec=15.0):
                boost *= 1.2
            # 人脸丢失 → 降低
            if bus.has_recent(SignalTypes.FACE_LOST, max_age_sec=5.0):
                boost *= 0.5

        if module_id == "emg":
            # 肌肉紧张说明在工作, EMG 信号更有参考价值
            if bus.has_recent(SignalTypes.MUSCLE_TENSION_HIGH, max_age_sec=15.0):
                boost *= 1.2

        return boost

    # ── 辅助 ──────────────────────────────────────────────

    def _quality_to_factor(self, quality: float) -> float:
        """quality → 权重因子。gate 以下为 0, gate 到 1 线性。"""
        gate = self.config.quality_gate
        if quality <= gate:
            return 0.0
        return (quality - gate) / (1.0 - gate)

    @staticmethod
    def _determine_source(weights: Dict[str, float]) -> str:
        active_ids = [k for k, v in weights.items() if v > 0.01]
        if not active_ids:
            return "none"
        if len(active_ids) == 1:
            return f"{active_ids[0]}_only"
        if set(active_ids) == {"emg", "vision"}:
            return "fused"
        return "multi_fused"

    def _build_empty(self, now: float) -> FusedReading:
        reading = FusedReading(
            stamina=None,
            state="unknown",
            module_weights={},
            source="none",
            module_readings={},
            alerts=[],
            timestamp=now,
        )
        self.latest = reading
        return reading

    @staticmethod
    def _find_stamina(module_id: str, active: List[ModuleReading]) -> Optional[float]:
        for r in active:
            if r.module_id == module_id:
                return r.stamina
        return None

    @staticmethod
    def _find_dimension(
        module_id: str, active: List[ModuleReading], dim: str,
    ) -> Optional[float]:
        for r in active:
            if r.module_id == module_id:
                return r.dimensions.get(dim)
        return None

    @staticmethod
    def _find_field(
        module_id: str, active: List[ModuleReading], field_name: str,
    ) -> Optional[float]:
        for r in active:
            if r.module_id == module_id:
                return getattr(r, field_name, None)
        return None

    @staticmethod
    def _find_metadata(
        module_id: str, active: List[ModuleReading], key: str,
    ) -> Optional[Any]:
        for r in active:
            if r.module_id == module_id:
                return r.metadata.get(key)
        return None


# ── 辅助 ─────────────────────────────────────────────────

def _stamina_to_state(stamina: float) -> str:
    if stamina > 60:
        return "focused"
    if stamina > 30:
        return "fading"
    return "depleted"
