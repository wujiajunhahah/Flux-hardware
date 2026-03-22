# FluxChi -- 开源工具链调研报告

> 调研日期: 2026-03-20
> 目标: 找到可直接替代手写阈值/规则的成熟开源工具，实现"模型替我们思考"

---

## 一、核心结论

**不要自己写 EAR/MAR 阈值判断。** 已有多个经过海量数据训练的开源模型可以直接输出我们需要的所有指标，
且精度远超手写规则。推荐组合:

```
BlendShapes (免费，已在栈内)  →  替代手算 EAR/MAR
OpenFace 3.0 (pip install)   →  替代手写 AU/gaze/emotion 检测
pyVHR (pip install)           →  新增 rPPG 心率维度
数据飞轮                      →  替代手设权重
```

---

## 二、工具详细评估

### 2.1 MediaPipe BlendShapes -- 最快胜利

- **是什么**: MediaPipe FaceLandmarker 内置的 52 个面部动作系数输出
- **已有基础**: 前端 vision-capture.js 已经在用 FaceLandmarker，只需 `outputFaceBlendshapes: true`
- **替代什么**: 手算 EAR (Eye Aspect Ratio) + 手算 MAR (Mouth Aspect Ratio) + 手写阈值

**关键 BlendShapes (52 个中与疲劳相关的):**

| BlendShape | 值域 | 替代 | 优势 |
|------------|------|------|------|
| `eyeBlinkLeft` | 0-1 | 手算 EAR (左眼) | Google 大规模数据训练，内置个人差异泛化 |
| `eyeBlinkRight` | 0-1 | 手算 EAR (右眼) | 同上 |
| `jawOpen` | 0-1 | 手算 MAR | 比 MAR 更鲁棒，不受嘴型/牙齿影响 |
| `eyeSquintLeft/Right` | 0-1 | 无 (新增) | 眯眼程度，疲劳辅助信号 |
| `eyeLookDownLeft/Right` | 0-1 | 手算 head pitch | 眼球下看，比头部 pitch 更精确 |
| `browDownLeft/Right` | 0-1 | 无 (新增) | 皱眉，压力/疲劳/专注指标 |
| `mouthFunnel` | 0-1 | 无 (新增) | 哈欠时嘴巴圆形张开 (区分说话) |

**接入成本**: 极低
- 前端: 改一行配置 + vision_frame 增加 blendshapes 字段
- 后端: VisionEngine 用 eyeBlink 替代手算 EAR，用 jawOpen 替代手算 MAR
- 不需要额外安装任何依赖

**局限**:
- 只有面部动作系数，没有 AU (Action Unit) 编码
- 没有注视方向 (gaze) 估计
- 没有情绪识别
- 在浏览器端运行，受浏览器后台限流影响

---

### 2.2 OpenFace 3.0 -- 最强单模型

- **是什么**: CMU MultiComp Lab 2025 年发布的面部行为分析工具包
- **论文**: "OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis" (FG 2025)
- **GitHub**: https://github.com/CMU-MultiComp-Lab/OpenFace-3.0
- **安装**: `pip install openface-test && openface download`
- **前代**: OpenFace 2.0 (C++, 难集成); 3.0 是 PyTorch 原生 Python 包

**一个模型同时输出:**

| 输出 | 说明 | 替代什么 |
|------|------|----------|
| **17 个 AU 强度** | FACS Action Unit 强度 (0-5) | BlendShapes (更细粒度) |
| **注视方向 (gaze)** | 左右眼的 3D gaze vector | 手算 head pose 分心判断 (更精确) |
| **6 种情绪** | happy/sad/angry/surprise/fear/disgust + neutral | 无 (新维度: boredom 可间接推断) |
| **68 landmarks** | 面部关键点 | MediaPipe 478 点 (精度相当) |

**关键 AU 与疲劳的关系:**

| AU | 名称 | 疲劳含义 |
|----|------|----------|
| AU45 | Blink | 眨眼 (直接替代 EAR 阈值) |
| AU43 | Eyes Closed | 闭眼 (PERCLOS 直接来源) |
| AU5 | Upper Lid Raiser | 睁大眼 (清醒时升高，疲劳时降低) |
| AU4 | Brow Lowerer | 皱眉 (专注/疲劳/压力) |
| AU27 | Mouth Stretch | 哈欠 (比 MAR 更准确) |
| AU26 | Jaw Drop | 下巴下落 (哈欠辅助) |
| AU9 | Nose Wrinkler | 皱鼻 (厌恶/不适) |

**架构:**
```
输入图像 → RetinaFace (人脸检测) → STAR (landmark) → 多任务头
                                                      ├→ AU 强度
                                                      ├→ Gaze 方向
                                                      └→ Emotion 分类
```

**性能 (论文数据):**
- 推理速度: ~30ms/帧 (CPU), ~10ms/帧 (GPU)
- AU 检测: F1 0.65+ (DISFA), 优于 OpenFace 2.0
- Gaze 估计: 误差 ~5.5 度 (MPIIFaceGaze)
- 非正面脸: 比 OpenFace 2.0 显著改善

**接入方式**: 后端 Python 侧运行，每 N 秒从摄像头帧抽样分析
- 不在浏览器运行 (PyTorch 模型太大)
- 前端仍用 MediaPipe 做实时 landmark + BlendShapes
- 后端用 OpenFace 3.0 做深层分析 (AU + gaze + emotion)

**局限**:
- 需要 GPU 才能达到实时帧率 (CPU ~30ms 可接受)
- 模型权重 ~300MB+
- 目前 2025 年 6 月刚发布，社区反馈尚少

---

### 2.3 py-feat -- 学术级分析工具箱

- **是什么**: Dartmouth COSAN Lab 开发的面部表情分析工具箱
- **论文**: "Py-Feat: Python Facial Expression Analysis Toolbox" (Affective Science, 2023)
- **GitHub**: https://github.com/cosanlab/py-feat
- **官网**: https://py-feat.org/
- **安装**: `pip install py-feat`
- **许可**: MIT

**功能:**
- 面部检测 (多种检测器可选)
- 68 landmark 检测
- AU 检测 (intensity + presence)
- 情绪识别 (7 类)
- HOG 特征提取
- **统计分析**: t-test, 回归, 被试间相关 (研究导向)
- **可视化**: 面部激活热力图, AU overlay

**与 OpenFace 3.0 比较:**

| 维度 | py-feat | OpenFace 3.0 |
|------|---------|--------------|
| 安装 | `pip install py-feat` | `pip install openface-test` |
| AU 检测 | 有 (多模型可选) | 有 (多任务统一模型) |
| Gaze | 无 | **有** |
| Emotion | 有 | 有 |
| 统计分析 | **有** (内置) | 无 |
| 推理速度 | 中等 | 更快 (统一模型) |
| 成熟度 | 2021 起，稳定 | 2025 年刚发布 |
| 适用场景 | 研究分析 | 实时应用 |

**判断**: FluxChi 是实时应用场景，OpenFace 3.0 更合适作为主力。
py-feat 可用于离线分析/飞轮训练阶段的特征提取。

---

### 2.4 pyVHR -- 摄像头心率提取 (rPPG)

- **是什么**: 远程光体积描记 (rPPG) 框架，从面部视频提取心率
- **论文**: "pyVHR: a Python framework for remote photoplethysmography" (PeerJ CS, 2022)
- **GitHub**: https://github.com/phuselab/pyVHR
- **安装**: conda + pip (有依赖)

**已实现的 rPPG 算法 (9 种经典 + 1 深度学习):**

| 算法 | 类型 | 说明 |
|------|------|------|
| **POS** | 信号处理 | Plane-Orthogonal-to-Skin, 最常用 |
| **CHROM** | 信号处理 | Chrominance-based, 光照鲁棒 |
| GREEN | 信号处理 | 绿色通道, 最简单 |
| ICA | 盲源分离 | 独立成分分析 |
| PCA | 降维 | 主成分分析 |
| SSR | 信号处理 | Spatial Subspace Rotation |
| LGI | 信号处理 | Local Group Invariance |
| PBV | 信号处理 | Plane-Based-on-Blood-Volume |
| OMIT | 信号处理 | Orthogonal Matrix Image Transformation |
| **MTTS-CAN** | 深度学习 | Multi-Task Temporal Shift CNN |

**对 FluxChi 的价值:**
- 心率变异性 (HRV) 下降是疲劳的**生理先兆**
- 比行为指标 (眨眼/哈欠) 早 5-10 分钟预警
- 前端已在采集 `skin.roiRgb` (额头 ROI RGB 均值)，这就是 rPPG 的原始输入
- 只需后端用 POS/CHROM 算法处理 RGB 时序即可

**接入方式:**
```
前端 (已有):  MediaPipe → 额头 ROI → RGB 均值 → vision_frame.skin.roiRgb
后端 (新增):  收集 30s RGB 序列 → pyVHR POS 算法 → HR (bpm) + HRV
             → HeartRateModule → ModuleReading → FusionEngine
```

**局限:**
- 需要 30+ 秒的稳定面部视频才能提取一次心率
- 对光照变化敏感 (办公室相对稳定，问题不大)
- conda 安装有些复杂

**替代方案: rPPG-Toolbox**
- GitHub: https://github.com/ubicomplab/rPPG-Toolbox
- NeurIPS 2023
- 更多深度学习模型 (PhysMamba, RhythmFormer)
- 支持 7 个数据集
- 比 pyVHR 更活跃维护

---

### 2.5 OpenDBM -- 临床级数字生物标志物

- **是什么**: AiCure 开发的开源数字生物标志物平台
- **GitHub**: https://github.com/AiCure/open_dbm
- **官网**: https://aicure.com/opendbm

**从视频/音频提取的指标:**
- 面部表情动力学 (基于 OpenFace 2.0)
- 声学生物标志物 (基调、能量、频谱)
- 语言生物标志物 (语速、停顿、词汇复杂度)
- 运动生物标志物 (面部颤动、眼球运动)

**适用场景:**
- 更适合临床研究 (抑郁、阿尔茨海默、精神分裂)
- 对 FluxChi 的办公疲劳场景略显"重型"
- 可作为 V3 参考，当飞轮数据量足够时

**判断**: 暂不接入，但其架构 (视频→多维生物标志物→分类器) 值得参考。

---

### 2.6 LibreFace -- 轻量替代

- **是什么**: WACV 2024 发布的面部表情分析工具包
- **论文**: "LibreFace: An Open-Source Toolkit for Deep Facial Expression Analysis"
- **安装**: `pip install libreface`
- **功能**: AU 检测 + AU 强度估计 + 表情识别
- **特点**: 比 OpenFace 2.0 更轻量，比 py-feat 更专注于深度学习

**判断**: OpenFace 3.0 已覆盖其所有功能且更强，暂不需要。

---

## 三、推荐组合方案

### V1 架构 (最小改动，最大收益)

```
浏览器端 (实时, 30fps):
  MediaPipe FaceLandmarker
    ├→ 478 landmarks (已有)
    ├→ 52 BlendShapes (新增: eyeBlink, jawOpen, eyeSquint, ...)
    ├→ transformation matrix → head pose (已有)
    └→ 额头 ROI RGB (已有, 供 rPPG)

后端 (低频, 每 2-5 秒):
  OpenFace 3.0
    ├→ 17 AU 强度 (替代规则引擎的疲劳子分)
    ├→ gaze 方向 (替代 head pose 分心判断)
    └→ emotion (新维度)

  pyVHR / rPPG-Toolbox
    └→ HR + HRV (从 ROI RGB 时序提取)

  EMG 手环
    └→ StaminaReading (不变)

FusionEngine:
  输入: BlendShapes + AU + gaze + emotion + HR/HRV + EMG
  决策: 层次化规则 (Phase 0-2) → 飞轮模型 (Phase 3+)
```

### 信号流时间线

```
t=0s    摄像头开启, MediaPipe 开始输出 BlendShapes
t=0.1s  每帧: eyeBlink/jawOpen → VisionEngine → PERCLOS/哈欠检测
t=2s    每 2s: 抽帧送 OpenFace 3.0 → AU/gaze/emotion → VisionModule
t=30s   首次: 30s RGB 序列 → pyVHR POS → HR (bpm)
t=60s   PERCLOS 窗口满 60s, 开始输出可靠值
t=300s  哈欠窗口满 5min, 频率统计可靠
t=∞     飞轮数据积累, 训练个人化模型
```

---

## 四、各工具对比矩阵

| 能力 | 手写规则 (当前) | BlendShapes | OpenFace 3.0 | pyVHR |
|------|----------------|-------------|--------------|-------|
| 眨眼检测 | EAR<0.21 | eyeBlink 0-1 | AU45 | - |
| PERCLOS | EAR 计算 | eyeBlink 计算 | AU43 计算 | - |
| 哈欠 | MAR>0.6 | jawOpen + mouthFunnel | AU27+AU26 | - |
| 注视/分心 | head yaw>30° | eyeLookDown/Up | **gaze vector** | - |
| 情绪 | 无 | 无 | **6 类** | - |
| 心率 | 无 | 无 | 无 | **HR + HRV** |
| 运行位置 | 后端 | 浏览器 | 后端 | 后端 |
| 安装成本 | 0 | 改 1 行 | pip install | conda |
| 数据训练 | 无 | Google 大规模 | CMU 多任务 | 多算法 |
| 个人差异 | 固定阈值 | 已泛化 | 已泛化 | 信号级 |

---

## 五、实施优先级

| Phase | 工具 | 耗时 | 收益 |
|-------|------|------|------|
| **0** | BlendShapes 启用 | 30 分钟 | 替代手算 EAR/MAR，立刻提升精度 |
| **1** | OpenFace 3.0 接入 | 半天 | 替代所有手写检测器，增加 gaze + emotion |
| **2** | pyVHR rPPG 接入 | 1 天 | 增加心率维度，提前预警 |
| **3** | 飞轮训练 | 持续 | 用数据替代所有手设权重 |

---

## 六、参考文献

### 工具
- [OpenFace 3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0) -- Hu, Mathur et al., FG 2025
- [OpenFace 2.0](https://github.com/TadasBaltrusaitis/OpenFace) -- Baltrusaitis et al., IEEE FG 2018
- [py-feat](https://github.com/cosanlab/py-feat) -- Cheong et al., Affective Science 2023
- [pyVHR](https://github.com/phuselab/pyVHR) -- Boccignone et al., PeerJ CS 2022
- [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) -- Liu et al., NeurIPS 2023
- [OpenDBM](https://github.com/AiCure/open_dbm) -- AiCure, 2020
- [LibreFace](https://pypi.org/project/libreface/) -- Chang et al., WACV 2024
- [MediaPipe BlendShapes](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) -- Google

### 疲劳检测文献
- Soukupova & Cech 2016, "Real-Time Eye Blink Detection using Facial Landmarks"
- Wierwille & Ellsworth 1994, PERCLOS 定义 (FHWA-RD-98-099)
- Dinges & Grace 1998, "PERCLOS: A valid psychophysiological measure of alertness"
- Schleicher et al. 2008, "Blinks and saccades as indicators of fatigue"
- Bentivoglio et al. 1997, "Analysis of blink rate patterns in normal subjects"
- Caffier et al. 2003, "Experimental evaluation of eye-blink parameters as a drowsiness measure"
- Abtahi et al. 2014, "YawDD: A Yawning Detection Dataset"
- Bergasa et al. 2006, "Real-Time System for Monitoring Driver Vigilance"
- McDonald & Bhagat 2009, 分层告警体系

### 数据集
- UTA-RLDD: 60 人 30h 视频, 3 级困倦 (UT Arlington)
- NTHU-DDD: 36 人, 5 类事件 (National Tsing Hua University)
- YawDD: 322 段视频, 哈欠/说话 (IEEE DataPort, CC BY)
- DROZY: 14 人多模态 + KSS 评分
- mEBAL: 38 人逐帧眨眼标注
- EyeBlink8: 8 段视频逐帧标注
