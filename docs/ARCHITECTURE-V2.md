# FluxChi V2 -- Modular Sensor Fusion Architecture

> 设计目标：可插拔模块 + 文献驱动阈值 + 数据飞轮 + 模块间协作

---

## 一、系统总览

```
                        +------------------+
                        |   Data Flywheel  |
                        |  (标注/训练/部署) |
                        +--------+---------+
                                 |
  ┌──────────┐  ┌──────────┐  ┌──v───────┐  ┌──────────┐
  │ EMG 手环 │  │ 摄像头   │  │ 键鼠活动 │  │ 心率手表 │  ...
  │ Module   │  │ Module   │  │ Module   │  │ Module   │
  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │              │              │              │
       │  ModuleReading (标准化接口)  │              │
       ▼              ▼              ▼              ▼
  ┌────────────────────────────────────────────────────┐
  │              Context Bus (模块间协作层)             │
  │  EMG 告诉 Vision: "肌肉紧张↑" → Vision 降低阈值   │
  │  Vision 告诉 Fusion: "打瞌睡" → 触发 EMG 高频采样  │
  └──────────────────────┬─────────────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────────┐
  │          Fusion Engine (N-source, quality-weighted)   │
  │  V1: 质量加权平均                                     │
  │  V2: 模型驱动 (data flywheel 产出的分类器)            │
  └──────────────────────┬───────────────────────────────┘
                         ▼
                   FusedReading
                   (stamina + state + alerts)
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        DecisionEngine        WebSocket broadcast
        (建议休息/继续)        (前端仪表盘)
```

---

## 二、文献驱动的阈值体系

### 2.1 EAR (Eye Aspect Ratio) -- 眨眼检测

| 参数 | 值 | 来源 |
|------|-----|------|
| 闭眼阈值 | **0.21** | Soukupova & Cech 2016, "Real-Time Eye Blink Detection using Facial Landmarks" -- 68-point dlib 模型下 EAR<0.2 为闭眼; MediaPipe 478-point 模型分辨率更高，经验修正为 0.21 |
| 连续帧数 | **3 帧** (@30fps = 100ms) | 同上论文; 正常眨眼持续 100-400ms (Schiffman 2001) |
| 长时间闭眼 | **>500ms (15帧@30fps)** | 区分眨眼与微睡眠 (microsleep); Wierwille & Ellsworth 1994 |

**关键洞察**: EAR 阈值高度依赖个人眼型和相机距离。文献中 0.19-0.25 均有报告。**正确做法是用前 30s 的个人基线标定**，而非固定阈值。

### 2.2 PERCLOS -- 金标准疲劳指标

| 参数 | 值 | 来源 |
|------|-----|------|
| 定义 | **P80**: 眼睑遮盖瞳孔 >= 80% 的时间比例 | Wierwille 1994; FHWA-RD-98-099 (NHTSA 官方报告) |
| 窗口 | **60 秒** 滑动窗口 | Dinges & Grace 1998, "PERCLOS: A valid psychophysiological measure of alertness" |
| 清醒 | **< 0.15** (15%) | FHWA-RD-98-099 |
| 轻度疲劳 | **0.15 - 0.25** | Dinges 1998; 对应 KSS 5-6 |
| 中度疲劳 | **0.25 - 0.40** | 同上; 对应 KSS 7-8 |
| 重度疲劳 | **> 0.40** | Wierwille: >0.40 时驾驶事故率显著上升 |

**注意**: 当前代码用 EAR<阈值 近似 P80。严格的 P80 需要估计眼睑开合度百分比，EAR 是合理近似但不完全等价。

### 2.3 眨眼率 (Blink Rate)

| 参数 | 值 | 来源 |
|------|-----|------|
| 正常静息 | **15-20 次/分钟** | Bentivoglio 1997, "Analysis of blink rate patterns in normal subjects" |
| 认知负荷高 | **下降到 6-10 次/分钟** | Stern 1994; 专注阅读/编程时眨眼减少 |
| 疲劳初期 | **升高到 25-30 次/分钟** | Schleicher 2008, "Blinks and saccades as indicators of fatigue" |
| 严重困倦 | **下降 + 闭眼时间延长** | Caffier 2003; 此时 PERCLOS 更可靠 |

**关键洞察**: 眨眼率与疲劳不是线性关系，而是**倒 U 型**:
- 专注 → 低频短眨眼
- 疲劳初期 → 高频
- 严重困倦 → 低频但闭眼时间长

当前代码只检测"过高"，遗漏了"异常低 + 长闭眼"的严重困倦模式。

### 2.4 MAR (Mouth Aspect Ratio) -- 哈欠检测

| 参数 | 值 | 来源 |
|------|-----|------|
| 哈欠阈值 | **0.50 - 0.65** | Abtahi 2014, "YawDD: A Yawning Detection Dataset"; 无标准化论文如 EAR，但 0.5-0.6 被多数工作采用 |
| 最短持续时间 | **2.0 - 4.0 秒** | Provine 1986: 平均哈欠持续约 6 秒; >2s 可区分说话 |
| 疲劳指示 | 5 分钟内 >= **3 次** | 经验值; Vural 2007 用类似窗口 |

**问题**: MAR 无法区分说话和哈欠。改进方案：
1. 哈欠时嘴巴**圆形张开**，说话时多为**水平张开** → 需要 aspect ratio + 嘴形分析
2. 哈欠持续时间明显长于说话的单个音节
3. **配合 EAR**: 哈欠时常伴随眯眼 (EAR 下降)

### 2.5 头部姿态 (Head Pose)

| 参数 | 值 | 来源 |
|------|-----|------|
| 打瞌睡 pitch | **> 15-20 度 (低头)** | Bergasa 2006, "Real-Time System for Monitoring Driver Vigilance" |
| 打瞌睡持续 | **> 2.0 秒** | 同上; 短暂低头可能是看手机/键盘 |
| 注意力分散 yaw | **> 30 度** | Tawari & Trivedi 2014, "Robust and Continuous Estimation of Driver Gaze Zone" |
| 分散时间比例 | 30s 窗口内 **> 25%** 时间偏转 | Murphy-Chutorian & Trivedi 2009 |

**办公场景修正**: 办公时低头看键盘是正常行为 (pitch 10-25 度)。驾驶场景的 15 度阈值太敏感。办公场景建议：
- 低头瞌睡: **pitch > 25 度 + 持续 > 3 秒 + EAR 偏低** (多维度联合判断)
- 注意力分散: **yaw > 35 度** (办公比驾驶允许更大的头部转动)

### 2.6 综合疲劳评分 -- 超越简单加权

当前代码的线性加权 `0.4*PERCLOS + 0.15*blink + 0.25*head + 0.2*yawn` 过于简化。文献中更好的方案：

**层次化决策 (推荐 V1):**
```
Level 3 (严重): PERCLOS > 0.30 OR 微睡眠 (闭眼>500ms) OR 连续点头
  → 立即告警, stamina 直降到 <15

Level 2 (中度): PERCLOS 0.15-0.30 OR 哈欠频繁 OR 眨眼率异常高
  → 建议休息, stamina 30-60

Level 1 (轻度): 眨眼率偏高 OR 偶尔走神 OR 偶尔哈欠
  → 提示注意, stamina 60-80

Level 0 (清醒): 无异常
  → stamina 80-100
```

来源: McDonald & Bhagat 2009 提出类似的分层告警体系; ISO 17488:2016 (Road vehicles -- Driver drowsiness) 也建议多级告警。

**Fuzzy Logic (推荐 V2):**
- 每个指标映射到模糊集 (低/中/高)
- 模糊规则: IF PERCLOS=高 AND 眨眼率=高 THEN fatigue=严重
- 比线性加权更能捕捉非线性关系
- 参考: Khushaba 2011, "Driver Drowsiness Classification Using Fuzzy Wavelet-Packet-Based Feature-Extraction Algorithm"

**学习型融合 (V3, 需要数据飞轮):**
- 轻量分类器 (XGBoost/小型 MLP) 在标注数据上训练
- 输入: 所有模块的 ModuleReading 特征向量
- 输出: fatigue_level (4 类) + stamina (回归)

---

## 三、可用的预训练模型和数据集

### 3.1 可直接使用的模型

| 模型 | 输入 | 输出 | 运行环境 | 来源 |
|------|------|------|----------|------|
| **MediaPipe Face Landmarker** | 视频帧 | 478 landmarks + transformation matrix | 浏览器 WASM/JS | 已在用 |
| **MediaPipe Face Mesh + BlendShapes** | 视频帧 | 52 个 blendshape 系数 (含 eyeBlinkLeft/Right, jawOpen 等) | 浏览器 WASM | Google; 比手算 EAR/MAR 更准确 |
| **pyVHR** | 面部视频 ROI | 心率 (rPPG) | Python (CPU) | Boccignone 2022; pip install pyvhr |
| **rPPG-Toolbox** | 面部视频 | HR, HRV | Python (GPU optional) | Liu 2023; GitHub: ubicomplab/rPPG-Toolbox |
| **ONNX ResNet-18 drowsiness** | 面部裁剪图 224x224 | drowsy/alert 二分类 | Python ONNX / 浏览器 ONNX.js | 多个 HuggingFace 社区模型; 基于 UTA-RLDD 训练 |

**BlendShapes 关键改进**: MediaPipe BlendShapes 输出 `eyeBlinkLeft` (0-1) 和 `jawOpen` (0-1)，比手算 EAR/MAR 更鲁棒：
- 已经过 Google 大规模数据训练
- 内置了个人差异的泛化能力
- 直接输出 0-1 归一化值，无需自己设阈值

### 3.2 公开数据集

| 数据集 | 规模 | 标注 | 获取 | 许可 |
|--------|------|------|------|------|
| **UTA-RLDD** | 60 人, 30h 视频 | 3 级困倦 (alert/low/high drowsiness) | 需申请; UT Arlington | 研究用 |
| **NTHU-DDD** | 36 人, 红外+可见光 | 5 类事件 (normal/yawning/slow-blink/nodding/looking-aside) | 需申请; NTHU | 研究用 |
| **YawDD** | 322 段视频 | 哈欠/正常/说话 | 公开; IEEE DataPort | CC BY |
| **ZJU Drowsiness** | 16 人, EEG+视频 | 连续困倦评分 | 需申请; 浙大 | 研究用 |
| **DROZY** | 14 人, 多模态 | KSS 主观评分 + 视频 + EEG | 公开; Massoz 2016 | 研究用 |
| **EyeBlink8** | 8 段视频 | 逐帧眨眼标注 | 公开; Drutarovsky & Fogelton | 研究用 |
| **mEBAL** | 38 人 | 逐帧眨眼标注 + NIR 多模态 | 公开; Hernandez-Ortega 2020 | CC BY |

### 3.3 推荐 V1 方案: BlendShapes 替代手算 EAR/MAR

```
当前:  MediaPipe landmarks → 手算 EAR/MAR → 手写阈值 → 判断
改进:  MediaPipe BlendShapes → eyeBlink/jawOpen (0-1) → 层次化决策 → 判断
```

**前端改动**: `vision-capture.js` 中 FaceLandmarker 已设 `outputFaceBlendshapes: false`，只需改为 `true`，然后在 vision_frame 中增加 `blendshapes` 字段。

---

## 四、模块化传感器框架

### 4.1 标准接口: ModuleReading + SensorModule

```python
# src/sensor_module.py

@dataclass
class ModuleReading:
    """所有传感器模块的标准化输出。"""
    module_id: str                  # "emg" / "vision" / "input_activity" / ...
    stamina: float                  # 0-100, 统一量纲
    quality: float                  # 0-1, 信号质量 (供融合降权)
    dimensions: Dict[str, float]    # 模块特有维度指标
    metadata: Dict[str, Any]        # 任意元数据
    alerts: List[str]               # 模块级告警
    timestamp: float

class SensorModule(ABC):
    """传感器模块抽象基类。"""
    module_id: str                  # 唯一标识
    display_name: str               # 人类可读名
    latest: Optional[ModuleReading] # 最近读数

    @abstractmethod
    def reset(self) -> None: ...

    def stale_seconds(self, now=None) -> float: ...
```

### 4.2 模块间协作: Context Bus

**核心思想**: 模块不只是独立打分，还可以互相通知上下文信息，影响彼此的行为。

```python
# src/context_bus.py

@dataclass
class ContextSignal:
    """模块间传递的上下文信号。"""
    source_module: str          # 发送方模块 ID
    signal_type: str            # 信号类型
    value: Any                  # 信号值
    timestamp: float
    ttl_sec: float = 10.0       # 生存时间

class ContextBus:
    """模块间的消息总线。模块可以发布信号，其他模块可以订阅。"""

    def publish(self, signal: ContextSignal) -> None: ...
    def query(self, signal_type: str, max_age_sec: float = 10.0) -> List[ContextSignal]: ...
```

**协作场景示例:**

| 发送方 | 信号 | 接收方 | 效果 |
|--------|------|--------|------|
| Vision | `microsleep_detected` | EMG | EMG 模块进入高灵敏度模式，降低 fatigue 判定阈值 |
| Vision | `perclos_rising` | Fusion | 提高 Vision 权重 (当前 Vision 比 EMG 更有信息量) |
| EMG | `muscle_tension_high` | Vision | Vision 降低"分心"的判定灵敏度 (肌肉紧张说明在工作) |
| EMG | `emg_signal_lost` | Fusion | 完全切到 Vision-only 模式 |
| InputActivity | `keyboard_active` | Vision | 低头 pitch 容忍度提高 (看键盘是正常的) |
| InputActivity | `idle_5min` | Vision | 提高所有疲劳指标的灵敏度 |
| HeartRate | `hrv_declining` | Fusion | 全局 fatigue bias +10% (自主神经系统疲劳先兆) |

### 4.3 N-Source Fusion Engine

```python
# src/fusion_engine.py (重构)

class FusionEngine:
    def __init__(self, config=None):
        self._modules: Dict[str, SensorModule] = {}
        self._context_bus = ContextBus()

    def register(self, module: SensorModule) -> None: ...
    def unregister(self, module_id: str) -> None: ...

    def fuse(self, now=None) -> FusedReading:
        """
        1. 收集所有模块 latest (过滤 stale + low quality)
        2. 查询 context_bus 获取模块间协作信号
        3. 动态调整权重 (基于 quality + context signals)
        4. 加权融合 stamina
        5. 层次化决策覆盖 (Level 3 事件直接降分)
        6. 应用告警惩罚
        7. 输出 FusedReading (含向后兼容字段)
        """

    @property
    def context_bus(self) -> ContextBus:
        return self._context_bus
```

### 4.4 向后兼容

所有现有前端 payload 字段保持不变，仅 **additive** 新增:
- `payload["modules"]`: `{module_id: reading_dict}` — 各模块原始读数
- `payload["fusion"]["module_weights"]`: `{module_id: weight}` — 实际权重明细
- `payload["fusion"]["source"]`: 新增 `"multi_fused"` 值 (3+ 源时)

---

## 五、数据飞轮 (Data Flywheel)

### 5.1 飞轮架构

```
  ┌───────────────────────────────────────────────────┐
  │                     用户使用系统                     │
  │  (EMG + 摄像头 + 键鼠活动 → 实时 ModuleReadings)   │
  └───────────┬───────────────────────────┬───────────┘
              │ 自动采集                   │ 主动标注
              ▼                           ▼
  ┌───────────────────┐     ┌──────────────────────┐
  │  原始特征日志       │     │  标注事件             │
  │  (每5s一条摘要)     │     │  "我现在很累" button  │
  │  vision dimensions │     │  "我很清醒" button    │
  │  emg dimensions    │     │  定时弹窗 KSS 1-9    │
  │  input events      │     │  自动: 休息前最后5min  │
  └───────────┬────────┘     └──────────┬───────────┘
              │                         │
              ▼                         ▼
  ┌──────────────────────────────────────────────────┐
  │              本地数据湖 (SQLite / Parquet)         │
  │  timestamp | module_readings_json | label | kss  │
  └───────────────────────┬──────────────────────────┘
                          │ 周期性训练 (本地/云端)
                          ▼
  ┌──────────────────────────────────────────────────┐
  │           个人化模型训练 (V3)                      │
  │  输入: 多模块特征向量 (EMG dims + Vision dims     │
  │        + input activity + context signals)        │
  │  输出: fatigue_level (4类) + stamina (回归)        │
  │  方法: XGBoost / 小型 MLP / fine-tune             │
  └───────────────────────┬──────────────────────────┘
                          │ 部署
                          ▼
  ┌──────────────────────────────────────────────────┐
  │       更新 Fusion Engine 的决策逻辑               │
  │  V1: 层次化规则 (文献阈值)                        │
  │  V2: 个人化阈值 (基线标定)                        │
  │  V3: 模型驱动 (替换规则引擎)                      │
  └──────────────────────────────────────────────────┘
```

### 5.2 数据采集策略

**自动采集 (零用户负担):**
- 每 5 秒存储一条 `snapshot`: 所有模块的 `dimensions` + `quality` + `alerts`
- 不存储原始视频/EMG 波形 (隐私 + 存储)
- 每次会话结束时生成 `session_summary.json`

**半自动标注 (低负担):**
- 仪表盘添加两个按钮: "我现在很累" / "我很清醒"
- 按下时标记最近 5 分钟的数据为对应标签
- 系统主动学习: 当模块间判断分歧大时 (EMG 说 OK, Vision 说疲劳)，弹窗问用户

**自动标签推断:**
- 用户主动点"休息"→ 标记之前 5 分钟为 "fatigued"
- 用户早晨开始工作 → 前 10 分钟标记为 "alert"
- 连续工作 >2 小时 → 后段倾向标记为 "fatigued" (弱标签)

### 5.3 基线标定 (V2, 替代固定阈值)

```python
class PersonalBaseline:
    """用户前 5 分钟的个人基线数据。"""

    def calibrate(self, readings: List[ModuleReading]) -> PersonalThresholds:
        """
        从清醒状态的数据中提取个人阈值:
        - ear_baseline: 用户正常睁眼时的 EAR 均值
        - ear_blink_threshold: ear_baseline * 0.75 (而非固定 0.21)
        - blink_rate_baseline: 用户正常眨眼频率
        - head_pitch_baseline: 用户正常工作姿势 pitch
        """
```

---

## 六、实施路线图

### Phase 0: P0 Bug Fixes (立即)
- [ ] 修复眨眼计数 bug (vision_engine.py)
- [ ] 修复无数据时 stamina=50 的误导 (fusion_engine.py)
- [ ] 启用 BlendShapes 输出 (vision-capture.js)

### Phase 1: 模块化框架 (1-2天)
- [ ] 新增 `src/sensor_module.py` (ModuleReading + SensorModule ABC)
- [ ] 新增 `src/modules/emg_module.py` (包装 StaminaEngine)
- [ ] 新增 `src/modules/vision_module.py` (包装 VisionEngine)
- [ ] 新增 `src/context_bus.py` (模块间消息总线)
- [ ] 重构 `src/fusion_engine.py` → N-source + ContextBus

### Phase 2: 文献驱动阈值 + 层次化决策 (1天)
- [ ] VisionEngine 阈值更新为文献值
- [ ] 实现层次化告警 (Level 0-3) 替代线性加权
- [ ] 添加 BlendShapes 路径 (eyeBlink/jawOpen 替代手算 EAR/MAR)
- [ ] 添加眨眼率"倒 U 型"检测

### Phase 3: 数据飞轮基础设施 (2天)
- [ ] 本地数据存储 (SQLite: snapshots 表 + labels 表)
- [ ] 前端添加 "我很累"/"我很清醒" 按钮
- [ ] 会话结束时自动生成 session_summary
- [ ] 自动标签推断逻辑

### Phase 4: 个人基线标定 (1天)
- [ ] 前 5 分钟基线采集流程
- [ ] 个人化阈值替换固定阈值
- [ ] 基线数据持久化

### Phase 5: 模型驱动融合 (数据充足后)
- [ ] 特征工程: ModuleReading → 特征向量
- [ ] XGBoost 分类器训练 pipeline
- [ ] ONNX 导出 + 服务端推理
- [ ] A/B: 规则引擎 vs 模型引擎

---

## 七、目录结构 (V2)

```
src/
  sensor_module.py            # NEW: ModuleReading, SensorModule ABC
  context_bus.py              # NEW: 模块间协作消息总线
  modules/
    __init__.py
    emg_module.py             # NEW: 包装 StaminaEngine
    vision_module.py          # NEW: 包装 VisionEngine
    input_activity_module.py  # FUTURE: 键鼠活动
    heartrate_module.py       # FUTURE: 心率手表
  fusion_engine.py            # REFACTORED: N-source + ContextBus
  energy.py                   # UNCHANGED
  vision_engine.py            # UPDATED: 文献阈值 + BlendShapes
  flywheel/
    __init__.py
    store.py                  # NEW: SQLite 数据存储
    labeler.py                # NEW: 自动/半自动标注
    baseline.py               # NEW: 个人基线标定
    trainer.py                # FUTURE: 模型训练 pipeline
  ...

web/
  app.py                      # MODIFIED: module registry 替换散落的引擎管理
  static/
    vision-capture.js         # UPDATED: BlendShapes 输出
    index.html                # UPDATED: 标注按钮, 简化 transport 区域
    vision.html               # UNCHANGED (调试页)
```
