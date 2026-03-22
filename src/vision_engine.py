"""
FluxChi -- Vision Engine
=========================================================
从浏览器 /ws/vision 推送的 vision_frame 中提取疲劳/注意力指标。

输入: vision_frame JSON (见 docs/VISION-WEBSOCKET-SPEC.md)
输出: VisionReading (PERCLOS, 眨眼率, 头姿偏离, 哈欠, 综合疲劳分)

设计原则:
  - 纯算法模块, 不含 WebSocket/HTTP 逻辑
  - 所有阈值可配置, 不硬编码
  - 信号质量 (quality) 始终伴随输出, 供 fusion 降权
=========================================================

头姿欧拉约定 (须与前端 MediaPipe 输出一致):
  - pitch: 抬头为正, 低头为负
  - yaw:   右转为正, 左转为负
  - roll:  右倾为正, 左倾为负
  首版疲劳检测仅使用绝对值, 减少正负约定歧义。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  配置 (所有阈值可调, 后续支持个人基线标定)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class VisionConfig:
    """可调阈值, 作为调参起点; 实际分布依赖相机距离/分辨率/个人眼型/眼镜。"""

    # -- 眨眼 / PERCLOS --
    ear_blink_threshold: float = 0.18       # EAR 低于此值视为闭眼
    ear_blink_consec_frames: int = 2        # 连续低于阈值帧数才算一次眨眼
    perclos_window_sec: float = 60.0        # PERCLOS 计算滑动窗口 (秒)
    perclos_mild: float = 0.15              # < 15%: 清醒
    perclos_moderate: float = 0.30          # 15-30%: 轻度疲劳; >30%: 严重疲劳

    # -- 眨眼率 --
    blink_rate_window_sec: float = 60.0     # 眨眼率统计窗口
    blink_rate_high: float = 25.0           # 次/分钟, 超过视为疲劳加剧

    # -- 头部姿态 --
    head_pitch_nod_deg: float = 20.0        # |pitch| 超过此值视为低头/打瞌睡
    head_pitch_nod_duration_sec: float = 3.0  # 持续超过此时间才报警
    head_yaw_distract_deg: float = 30.0     # |yaw| 超过此值视为注意力分散
    head_pose_window_sec: float = 30.0      # 头姿统计滑动窗口

    # -- 打哈欠 --
    mar_yawn_threshold: float = 0.6         # MAR 超过此值视为可能打哈欠
    mar_yawn_min_duration_sec: float = 1.5  # 持续超过此时间确认打哈欠
    yawn_window_sec: float = 300.0          # 哈欠频率统计窗口 (5 分钟)
    yawn_fatigue_count: int = 3             # 窗口内超过此数视为中度疲劳

    # -- 人脸丢失 --
    face_lost_timeout_sec: float = 5.0      # 连续无脸超过此时间, quality 降为 0

    # -- BlendShapes 模式 --
    # 有 blendshapes 时优先使用 (Google 大规模训练, 比几何 EAR/MAR 更鲁棒)
    # 无 blendshapes 时自动回退几何 EAR/MAR
    use_blendshapes_for_blink: bool = True  # eyeBlinkLeft/Right 替代 EAR
    use_blendshapes_for_yawn: bool = True   # jawOpen + mouthFunnel 替代 MAR
    bs_blink_threshold: float = 0.55        # eyeBlink > 此值视为闭眼 (BlendShapes 0-1)
    bs_blink_consec_frames: int = 2         # 连续闭眼帧数
    bs_yawn_jaw_threshold: float = 0.45     # jawOpen > 此值 (BlendShapes 模式)
    bs_yawn_funnel_threshold: float = 0.3   # mouthFunnel > 此值辅助确认 (区分说话)
    bs_yawn_min_duration_sec: float = 2.0   # 哈欠最短持续 (BlendShapes 模式, 比几何模式更严格)

    # -- 总分权重 (P0: 固定权重; P1: 动态) --
    w_perclos: float = 0.40
    w_blink_rate: float = 0.15
    w_head_pose: float = 0.25
    w_yawn: float = 0.20


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  输出数据结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class VisionReading:
    """VisionEngine 每次 update() 输出的快照。"""

    # -- 指标 --
    perclos: float              # 0-1, 窗口内眼睛闭合比例
    blink_rate: float           # 次/分钟
    blink_count_window: int     # 窗口内眨眼总次数

    head_pitch_mean: float      # 度, 窗口内 |pitch| 均值
    head_yaw_mean: float        # 度, 窗口内 |yaw| 均值
    head_nod_detected: bool     # 是否检测到低头打瞌睡
    head_distracted: bool       # 是否频繁分心 (|yaw| 超阈值)

    yawn_count_window: int      # 窗口内哈欠次数
    yawn_active: bool           # 当前正在打哈欠

    # -- 综合 --
    fatigue_score: float        # 0-1, 综合疲劳分 (越高越疲劳)
    alertness_level: str        # "alert" / "mild_fatigue" / "moderate_fatigue" / "severe_fatigue"
    quality: float              # 0-1, 信号质量 (人脸检出率 + 置信度)

    # -- 元数据 --
    face_present: bool
    frames_processed: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perclos": round(self.perclos, 3),
            "blink_rate": round(self.blink_rate, 1),
            "blink_count": self.blink_count_window,
            "head_pitch_mean": round(self.head_pitch_mean, 1),
            "head_yaw_mean": round(self.head_yaw_mean, 1),
            "head_nod": self.head_nod_detected,
            "head_distracted": self.head_distracted,
            "yawn_count": self.yawn_count_window,
            "yawn_active": self.yawn_active,
            "fatigue_score": round(self.fatigue_score, 3),
            "alertness": self.alertness_level,
            "quality": round(self.quality, 2),
            "face_present": self.face_present,
            "frames": self.frames_processed,
            "timestamp": self.timestamp,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  带时间戳的环形缓冲
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _TimedRingBuffer:
    """保留最近 `window_sec` 秒内的 (timestamp, value) 对。"""

    def __init__(self, window_sec: float):
        self.window_sec = window_sec
        self._buf: Deque[Tuple[float, Any]] = deque()

    def append(self, t: float, value: Any) -> None:
        self._buf.append((t, value))
        self._evict(t)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def values(self, now: Optional[float] = None) -> List[Any]:
        if now is not None:
            self._evict(now)
        return [v for _, v in self._buf]

    def count(self, now: Optional[float] = None) -> int:
        if now is not None:
            self._evict(now)
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  核心引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VisionEngine:
    """
    处理 vision_frame 流, 维护滑动窗口统计, 输出 VisionReading。

    用法:
        engine = VisionEngine()
        reading = engine.update(frame_dict)   # 每帧调用
        if reading:
            print(reading.fatigue_score)
    """

    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()
        self._total_frames = 0
        self._last_seq = -1

        # -- 眨眼状态机 --
        self._ear_below_count = 0           # 连续低于阈值的帧数
        self._in_blink = False              # 当前是否处于眨眼中
        self._blinks = _TimedRingBuffer(self.config.blink_rate_window_sec)

        # -- PERCLOS --
        self._eye_states = _TimedRingBuffer(self.config.perclos_window_sec)
        # 存储 (t, is_closed: bool)

        # -- 头姿 --
        self._head_pitches = _TimedRingBuffer(self.config.head_pose_window_sec)
        self._head_yaws = _TimedRingBuffer(self.config.head_pose_window_sec)
        self._pitch_nod_start: Optional[float] = None  # 低头开始时间

        # -- 打哈欠状态机 --
        self._mar_above_start: Optional[float] = None   # MAR 超阈值开始时间
        self._in_yawn = False
        self._yawns = _TimedRingBuffer(self.config.yawn_window_sec)

        # -- 人脸追踪 --
        self._last_face_seen: Optional[float] = None
        self._face_present_buf = _TimedRingBuffer(30.0)  # 30s 窗口算 quality

        # -- 输出缓存 --
        self.latest: Optional[VisionReading] = None

    def update(self, frame: Dict[str, Any]) -> Optional[VisionReading]:
        """
        处理一条 vision_frame, 返回 VisionReading。

        参数:
            frame: 符合 VISION-WEBSOCKET-SPEC.md schema:1 的 dict

        返回:
            VisionReading, 或 None (如果帧格式有误)
        """
        if frame.get("type") != "vision_frame":
            return None

        t = frame.get("t")
        if t is None:
            return None

        seq = frame.get("seq", 0)
        self._total_frames += 1
        self._last_seq = seq

        face = frame.get("face") or {}
        face_present = bool(face.get("present", False))
        face_confidence = float(face.get("confidence", 0.0))

        # 更新人脸追踪
        self._face_present_buf.append(t, face_present)
        if face_present:
            self._last_face_seen = t

        geometry = frame.get("geometry") or {}
        blendshapes = frame.get("blendshapes") or {}

        # -- 处理各维度 (blendshapes 优先, 几何回退) --
        self._process_blink(t, geometry, blendshapes, face_present)
        self._process_head_pose(t, geometry, face_present)
        self._process_yawn(t, geometry, blendshapes, face_present)

        # -- 计算输出 --
        reading = self._compute_reading(t, face_present, face_confidence)
        self.latest = reading
        return reading

    # ── 眨眼 / PERCLOS ──────────────────────────────────────

    def _process_blink(self, t: float, geo: Dict, bs: Dict, face_present: bool) -> None:
        """眨眼/PERCLOS 处理。BlendShapes 优先, 几何 EAR 回退。"""
        cfg = self.config

        if not face_present:
            self._ear_below_count = 0
            self._in_blink = False
            return

        # ── 判定闭眼: BlendShapes 优先 ──
        is_closed: Optional[bool] = None

        if cfg.use_blendshapes_for_blink and "eyeBlinkLeft" in bs and "eyeBlinkRight" in bs:
            # BlendShapes 路径: eyeBlink 0-1, 越高越闭合
            eye_blink = max(bs["eyeBlinkLeft"], bs["eyeBlinkRight"])
            is_closed = eye_blink > cfg.bs_blink_threshold
            consec_threshold = cfg.bs_blink_consec_frames
        else:
            # 几何回退: EAR 越低越闭合
            ear_left = geo.get("earLeft")
            ear_right = geo.get("earRight")
            ear = geo.get("ear")
            if ear is None:
                if ear_left is not None and ear_right is not None:
                    ear = min(ear_left, ear_right)
                else:
                    return
            is_closed = ear < cfg.ear_blink_threshold
            consec_threshold = cfg.ear_blink_consec_frames

        self._eye_states.append(t, is_closed)

        # ── 眨眼状态机 (与来源无关) ──
        if is_closed:
            self._ear_below_count += 1
            if self._ear_below_count >= consec_threshold and not self._in_blink:
                self._in_blink = True
        else:
            if self._in_blink:
                self._blinks.append(t, True)
                self._in_blink = False
            self._ear_below_count = 0

    def _compute_perclos(self, now: float) -> float:
        states = self._eye_states.values(now)
        if not states:
            return 0.0
        closed_count = sum(1 for s in states if s)
        return closed_count / len(states)

    def _compute_blink_rate(self, now: float) -> Tuple[float, int]:
        count = self._blinks.count(now)
        window = self.config.blink_rate_window_sec
        rate = count / (window / 60.0)  # 次/分钟
        return rate, count

    # ── 头部姿态 ─────────────────────────────────────────────

    def _process_head_pose(self, t: float, geo: Dict, face_present: bool) -> None:
        if not face_present:
            self._pitch_nod_start = None
            return

        pose = geo.get("headPoseDeg")
        if not pose:
            return

        pitch = pose.get("pitch", 0.0)
        yaw = pose.get("yaw", 0.0)

        self._head_pitches.append(t, abs(pitch))
        self._head_yaws.append(t, abs(yaw))

        # 低头打瞌睡检测
        if abs(pitch) > self.config.head_pitch_nod_deg:
            if self._pitch_nod_start is None:
                self._pitch_nod_start = t
        else:
            self._pitch_nod_start = None

    def _is_nodding(self, now: float) -> bool:
        if self._pitch_nod_start is None:
            return False
        return (now - self._pitch_nod_start) >= self.config.head_pitch_nod_duration_sec

    def _is_distracted(self, now: float) -> bool:
        """最近窗口内 |yaw| 超阈值的帧占比 > 30%。"""
        yaws = self._head_yaws.values(now)
        if not yaws:
            return False
        distracted_count = sum(1 for y in yaws if y > self.config.head_yaw_distract_deg)
        return (distracted_count / len(yaws)) > 0.3

    # ── 打哈欠 ───────────────────────────────────────────────

    def _process_yawn(self, t: float, geo: Dict, bs: Dict, face_present: bool) -> None:
        """哈欠检测。BlendShapes 优先 (jawOpen + mouthFunnel), 几何 MAR 回退。"""
        cfg = self.config

        if not face_present:
            self._mar_above_start = None
            self._in_yawn = False
            return

        # ── 判定嘴巴大张: BlendShapes 优先 ──
        mouth_open = False
        min_duration = cfg.mar_yawn_min_duration_sec

        if cfg.use_blendshapes_for_yawn and "jawOpen" in bs:
            # BlendShapes 路径: jawOpen 0-1 + mouthFunnel 辅助区分说话
            jaw = bs["jawOpen"]
            funnel = bs.get("mouthFunnel", 0.0)
            # 哈欠: 嘴巴大张 + 嘴型偏圆 (说话时 funnel 低)
            mouth_open = jaw > cfg.bs_yawn_jaw_threshold and funnel > cfg.bs_yawn_funnel_threshold
            min_duration = cfg.bs_yawn_min_duration_sec
        else:
            # 几何回退
            mar = geo.get("mar")
            if mar is None:
                return
            mouth_open = mar > cfg.mar_yawn_threshold

        if mouth_open:
            if self._mar_above_start is None:
                self._mar_above_start = t
            elif (t - self._mar_above_start) >= min_duration:
                if not self._in_yawn:
                    self._in_yawn = True
                    self._yawns.append(t, True)
        else:
            self._mar_above_start = None
            self._in_yawn = False

    # ── 综合计算 ─────────────────────────────────────────────

    def _compute_quality(self, now: float, face_confidence: float) -> float:
        """
        信号质量 0-1, 基于:
          - 近 30s 人脸检出率
          - 当前帧置信度
          - 距上次看到人脸的时间
        """
        face_states = self._face_present_buf.values(now)
        if not face_states:
            return 0.0

        detection_rate = sum(1 for s in face_states if s) / len(face_states)

        # 超时衰减
        timeout_factor = 1.0
        if self._last_face_seen is not None:
            elapsed = now - self._last_face_seen
            if elapsed > self.config.face_lost_timeout_sec:
                timeout_factor = 0.0
            elif elapsed > 1.0:
                timeout_factor = max(0.0, 1.0 - elapsed / self.config.face_lost_timeout_sec)

        return min(1.0, detection_rate * 0.5 + face_confidence * 0.3 + timeout_factor * 0.2)

    def _compute_reading(self, now: float, face_present: bool,
                         face_confidence: float) -> VisionReading:
        cfg = self.config

        perclos = self._compute_perclos(now)
        blink_rate, blink_count = self._compute_blink_rate(now)
        nod = self._is_nodding(now)
        distracted = self._is_distracted(now)
        yawn_count = self._yawns.count(now)

        pitches = self._head_pitches.values(now)
        yaws = self._head_yaws.values(now)
        pitch_mean = sum(pitches) / len(pitches) if pitches else 0.0
        yaw_mean = sum(yaws) / len(yaws) if yaws else 0.0

        quality = self._compute_quality(now, face_confidence)

        # ── 各维度疲劳子分 (0-1, 越高越疲劳) ──

        # PERCLOS 子分: 线性映射 [0, 0.5] -> [0, 1]
        perclos_score = min(1.0, perclos / 0.5)

        # 眨眼率子分: 正常 12-20 次/min; >25 视为疲劳
        if blink_rate <= 15:
            blink_score = 0.0
        elif blink_rate <= cfg.blink_rate_high:
            blink_score = (blink_rate - 15) / (cfg.blink_rate_high - 15)
        else:
            blink_score = min(1.0, 0.5 + (blink_rate - cfg.blink_rate_high) / 20)

        # 头姿子分: nod = 高疲劳, distracted = 中等
        head_score = 0.0
        if nod:
            head_score = 1.0
        elif distracted:
            head_score = 0.6
        else:
            # 基于 pitch 偏离: |pitch| > 10 开始累计
            head_score = min(1.0, max(0.0, (pitch_mean - 10) / 20))

        # 哈欠子分
        yawn_score = min(1.0, yawn_count / max(1, cfg.yawn_fatigue_count))

        # ── 加权融合 ──
        fatigue_score = (
            cfg.w_perclos * perclos_score
            + cfg.w_blink_rate * blink_score
            + cfg.w_head_pose * head_score
            + cfg.w_yawn * yawn_score
        )
        fatigue_score = max(0.0, min(1.0, fatigue_score))

        # ── 等级判断 ──
        if fatigue_score < 0.2:
            level = "alert"
        elif fatigue_score < 0.45:
            level = "mild_fatigue"
        elif fatigue_score < 0.7:
            level = "moderate_fatigue"
        else:
            level = "severe_fatigue"

        return VisionReading(
            perclos=perclos,
            blink_rate=blink_rate,
            blink_count_window=blink_count,
            head_pitch_mean=pitch_mean,
            head_yaw_mean=yaw_mean,
            head_nod_detected=nod,
            head_distracted=distracted,
            yawn_count_window=yawn_count,
            yawn_active=self._in_yawn,
            fatigue_score=fatigue_score,
            alertness_level=level,
            quality=quality,
            face_present=face_present,
            frames_processed=self._total_frames,
            timestamp=now,
        )

    # ── 外部接口 ─────────────────────────────────────────────

    def reset(self) -> None:
        """重置所有状态 (如用户手动校准)。"""
        self._total_frames = 0
        self._last_seq = -1
        self._ear_below_count = 0
        self._in_blink = False
        self._blinks.clear()
        self._eye_states.clear()
        self._head_pitches.clear()
        self._head_yaws.clear()
        self._pitch_nod_start = None
        self._mar_above_start = None
        self._in_yawn = False
        self._yawns.clear()
        self._last_face_seen = None
        self._face_present_buf.clear()
        self.latest = None

    @property
    def dropped_frames(self) -> int:
        """粗略估计: seq 不连续的次数 (不含首帧)。"""
        # 需要在 update 中追踪, 此处简化
        return 0
