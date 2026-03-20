# FluxChi × LeLamp：HCI / HRI 设计空间与对接备忘

> 不接硬件也可推进：设计空间、用户叙事、研究方法与「桥接层」约定。  
> LeLamp 硬件与软件以 [Human Computer Lab — LeLamp](https://github.com/humancomputerlab/LeLamp) / [lelamp_runtime](https://github.com/humancomputerlab/lelamp_runtime) 为准；官网 [lelamp.com](https://www.lelamp.com/) 强调本地可控、云可选。

---

## 1. 一页设计空间表（StaminaState × 渠道 × 社会性含义）

**约定**：下列「含义」是**设计意图**（intended social meaning），实现时需与文案、动画强度一致，避免「数据说疲惫、机器人却在庆祝」的语义冲突。

| 离散态（引擎） | App（视觉 + 文案角色） | 通知（打断性） | LeLamp · 光 | LeLamp · 动 | LeLamp · 声 |  intended social meaning（跨渠道一致） |
|----------------|------------------------|----------------|-------------|-------------|-------------|----------------------------------------|
| **focused** | 高饱和续航环/主色；文案偏「状态确认」而非表扬 | 默认不推送；除非用户开启「深度工作守护」 | 稳定亮度或轻微呼吸；避免炫彩 | 静止或极低幅「就绪」姿态（若有） | 无或极轻环境音 | **「你正处于可维持区间」** — 镜像当前资源，不施加道德评价。 |
| **fading** | 环色转警示；短文案提示「可准备微休息」 | 可延迟聚合推送（避免每 30s 一条） | 色温略暖/亮度缓变（attention without alarm） | 慢速、小幅度动作（若有） | 可选低频提示音（可关） | **「负荷在累积，但仍在可控范围」** — 预警而非命令。 |
| **depleted** | 明确低续航；CTA 指向休息/分段 | 允许较强提醒（用户可控级别） | 明显但非频闪的暗/红策略（注意光敏） | 避免欢快动作；可做「低头/静止」类收敛 | 可选单次柔和 chime（可关） | **「需要恢复」** — 严肃但不羞辱，强调身体信号而非「你不够努力」。 |
| **recovering** | 绿色/恢复曲线；文案强调回升与节奏 | 少打扰；可「恢复达标」时一条正向摘要 | 渐亮、偏冷或自然白光感 | 缓慢舒展类预设（若有 replay） | 无或极轻 | **「系统在回到可用状态」** — 与专注态区分，避免与 focused 同义重复。 |
| **（建议扩展）resting / 休息模式** | App 内全屏休息 UI（你已用深绿背景） | 一般静默 | 显著降亮度、低刺激 | 停止表达性动作或极慢节拍 | 无 | **「允许暂停」** — 与工作状态明确分相，降低「偷懒」污名。 |
| **（建议扩展）disconnected** | 显示连接引导 | 可选「已断开」一条 | 中性灰或呼吸「待机」 | 静止 | 无 | **「输入暂不可用」** — 不暗示用户过错（BLE/USB/网络问题）。 |

**与 FluxChi 代码对齐**：主应用内 `StaminaState` 为 `focused | fading | depleted | recovering`（见 `FluxModels.swift`）；休息/断连可作为 **映射层的扩展枚举**（例如 `FluxAmbientPhase`），再一一落到上表。

---

## 2. 用户叙事二选一（及 HRI 信任模型差异）

### 叙事 A：**环境镜像（Ambient Mirror）**

- **定义**：LeLamp（及 App）主要**反映**当前生理—工作综合状态（stamina / 状态机），**不扮演「督促你」的他者**。
- **用户心里模型**：「灯/小机器人在显示我的身体节奏」，类似温度计，但带一点生命感。
- **信任特征**：更依赖 **校准透明、可解释、可关闭**；用户对「误判」容忍度略高，但若误判会削弱「镜像」信任。
- **风险**：长期可能被体验为「监视器」若缺少 **agency（关、降级、仅光不动）**。

### 叙事 B：**教练代理（Coach Agent）**

- **定义**：LeLamp **代表某种规范或建议**（何时停、如何恢复），语言/动作带「劝导」。
- **用户心里模型**：「它在管我 / 帮我」— 社会角色更强。
- **信任特征**：更依赖 **能力感知（competence）** 与 **善意（benevolence）**；误判代价更高（易引发抵触）。
- **风险**：与 EMG 强绑定易被读成 **绩效监控**；与「可爱伙伴」人设张力更大。

### 建议写进论文/产品的「一句话立场」

> **v1 默认采用叙事 A（环境镜像）**，通知与 Lamp 的「强干预」仅在 `depleted` 或用户显式开启「教练模式」时升级到叙事 B 的子集。  
> 这样 HRI 上保留 **低社会权力距离**，HCI 上仍保留 **可配置的打断梯度**。

（若你最终决定纯 B，需单独一节讨论 **权力动态、工作场景伦理、用户可控性**。）

---

## 2.5 与 Apple **ELEGNT** 对齐：把「打动你的那篇论文」接进 FluxChi + LeLamp

论文：[**ELEGNT: Expressive and Functional Movement Design for Non-anthropomorphic Robot**](https://arxiv.org/abs/2501.12493)（[PDF](https://arxiv.org/pdf/2501.12493)）。核心主张是：非人形（如灯具）机器人的运动设计应**同时**优化 **功能性**（任务、效率、安全）与 **表达性**（意图、注意、情绪等非言语线索）；用户研究中 **表达驱动** 的运动在社交向任务里更能提升参与感与对机器人的积极感知。同思路在 Apple ML 研究页有延续表述：[Elegnt — Expressive and Functional Movement](https://machinelearning.apple.com/research/elegnt-expressive-functional-movement)。[LeLamp 硬件 README](https://github.com/humancomputerlab/LeLamp) 亦明确基于该研究脉络——你与 LeLamp 的结合，天然可以用 **ELEGNT 当理论骨架**，避免只做「彩色台灯联动」。

### 2.5.1 与本文 §1 设计空间表的关系（无矛盾，而是分层）

| ELEGNT 侧 | 在本文中的落点 |
|-----------|----------------|
| **Expressive utility**（表达效用） | 表 1 最后一列 **intended social meaning**，以及 LeLamp **光/动/声** 给观众的情绪与注意解读。 |
| **Functional utility**（功能效用） | **安全与可持续**：runtime 文档提醒长时间高负荷可能伤舵机；**不打断深度工作**：动作幅度、频率上限；**与 App 语义一致**：避免错误表达。 |
| **Movement generation / 原语** | 将 `StaminaState`（及扩展 `FluxAmbientPhase`）映射到 **录制 replay 名 + LED 档**（§3.3 / 预设 YAML），每条映射显式写 **双栏卡**（下表）。 |

### 2.5.2 建议：为每个状态填一张「双效用运动卡」（不接硬件也能写）

复制下表到笔记或 `docs/fluxchi_elegnt_motion_cards.md`；硬件到后把 `replay` / `led` 换成真实资源名。

| StaminaState | Functional（必须满足） | Expressive（ELEGNT 意图） |
|--------------|------------------------|-------------------------|
| **focused** | 低占空比动作；避免周期性大摆幅 | 「在场但不打扰」— 注意指向工作区而非用户脸部施压 |
| **fading** | 单次动作时长有上限；可合并 App 侧防抖 | 「温和提醒」— 累积负荷可见，非问责 |
| **depleted** | 强提醒仍避频闪与电机过热 | 「需要恢复」— 严肃、收敛，避免欢快拟人 |
| **recovering** | 与 depleted 动作库互斥，防语义混淆 | 「松弛与回升」— 慢节奏、开放姿态（若用 replay） |
| **resting** | 默认可关闭电机表达，仅光或全静 | 「许可暂停」— 与 A 叙事「环境镜像」一致 |
| **disconnected** | 安全默认位 / 关灯 | 「输入缺失」— 中性，不拟人化道歉过度 |

### 2.5.3 你可声明的「结合点 / contribution 方向」（写 CHI 好用）

1. **闭环输入**：ELEGNT 原文以场景 storyboard 与用户对比为主；FluxChi 提供 **连续生理—行为状态流**（EMG → StaminaEngine → 离散态），把 **内部状态 → 非人形表达** 封成可复现管道。  
2. **双叙事对照**：在 **镜像（§2A）** 下，表达性更偏「状态外化」；在 **教练（§2B）** 下，同一 `depleted` 可略增强「劝导性」动作幅度——正好对应论文里 **function- vs expression-driven** 的对照实验逻辑，可做成 **被试内 2×2**。  
3. **与 LeLamp 开源栈对齐**：用 **record/replay** 落地「表达原语」，用 **YAML 映射** 做版本化，论文可复现、工程不绑死 HTTP。

### 2.5.4 注意边界（审稿友好）

- **不要过度声称**「改进健康结局」：ELEGNT 与你的工作都更贴近 **感知、参与、信任**；生理信号用于 **工作情境介入** 需在伦理与局限中写清。  
- **Apple 论文 ≠ 产品授权**：引用 arXiv / 研究页即可；实现落在 **自研映射 + LeLamp 开源 runtime**。

---

## 3. Discord / hobbyist 文档与「真实 API」— 当前公开信息 vs 待确认清单

### 3.1 公开仓库已能确定的事实（截至文档编写时）

| 维度 | 公开信息 | 对 FluxChi 的含义 |
|------|-----------|-------------------|
| **控制栈** | [lelamp_runtime](https://github.com/humancomputerlab/lelamp_runtime)：**Python**、`uv`、入口含 `main.py` / `smooth_animation.py`；与 **串口舵机**（`lerobot-find-port`）、**录制/回放 CSV**（`lelamp.record` / `lelamp.replay`）、**RGB 测试**（`test_rgb`）等 | 没有现成「官方 HTTP 一键 API」时，**最稳的桥接**是在 **树莓派/运行 runtime 的机器** 上挂一个**薄网关**（见下节 `LeLampSink`）。 |
| **操作入口** | [LeLamp Control 教程](https://github.com/humancomputerlab/LeLamp/blob/master/docs/5.%20LeLamp%20Control.md)：SSH 到灯侧 `lelamp_runtime` 执行 `uv run -m lelamp.*` | 典型拓扑：**局域网内 SSH 或同机进程**；延迟取决于网络 + 回放序列长度，需实测。 |
| **语音/云** | Runtime README：`livekit-agents`、`.env` 中 LiveKit / OpenAI；**可选云** | 与 FluxChi **本地生理数据** 结合时，建议 **默认不走云**，或明确 **opt-in** 与数据最小化。 |
| **本地** | [官网 FAQ](https://www.lelamp.com/about)：可本地控制、云功能可关 | 与 FluxChi「手机/本机推理」叙事一致，适合论文 **privacy stance**。 |

### 3.2 需在 Discord / 社群中向维护者确认的问题（复制即用）

社区入口以官网为准：[Discord（lelamp.com 链接）](https://discord.gg/nJP6h8GhFw)；仓库文档中亦出现另一 invite（如 [Control 文档内链接](https://discord.gg/4hmNW3Ep)），**以社群当前置顶/公告为准**。

建议在 Discord 新开帖，直接问：

1. **是否有计划提供稳定的局域网 RPC**（HTTP/WebSocket/MQTT）用于触发「预设动作 / LED 模式」？还是推荐长期 SSH + CLI？
2. **从「收到命令」到「舵机/LED 开始执行」的典型端到端延迟**（Pi 本地 vs 从 Mac 经 SSH）？
3. **多客户端并发**（手机 + 电脑）时官方推荐架构？
4. **录制 replay 名称** 是否适合作为「表情原子」（与下表 `presetId` 对应）？有无版本字段？

### 3.3 `LeLampSink` 与公开栈的映射（建议实现形态）

在 FluxChi 侧先定 **与硬件无关** 的 sink 接口，便于 CHI 实验换实现：

```text
LeLampSink.apply(phase: FluxAmbientPhase, options: SinkOptions)
```

| Sink 实现 | 适用阶段 | 做法 |
|-----------|----------|------|
| **NullSink** | 无硬件 | 空操作，单元测试与对照实验。 |
| **HttpReplaySink**（自建） | 有 Pi / 同网 | 在 `lelamp_runtime` 同机跑 **极小 Flask/FastAPI**：接收 `POST /express { "preset": "recovering_calm", "led_profile": "..." }`，内部 `subprocess` 调 `uv run -m lelamp.replay ...` 或调用 Python API（若后续暴露）。 |
| **SshReplaySink**（原型） | 仅实验室 | 从 Mac `ssh pi@...` 执行一条 replay；延迟高、脆弱，仅 WoZ 前期。 |

**重要**：公开文档以 **CLI 录制/回放** 为主；**将生理状态映射到 `preset` 名称** 属于你的产品层约定，需版本化（例如 `fluxchi_lelamp_presets_v1.yaml`）。

---

## 4. 若走 CHI：硬件未到也可做的感知研究

### 4.1 Wizard-of-Oz（推荐）

- **做法**：被试佩戴手环 + 使用 FluxChi App；**LeLamp 由研究员手动触发** 与 App 同步的「光/动」（或预先录好的视频投影到灯位占位物）。
- **能回答**：社会含义是否被读成镜像 vs 教练、是否「监视感」、打断是否可接受。
- **与上表关系**：直接验证 **intended social meaning** 是否传到用户解释模型。

### 4.2 视频/交互原型（高保真低成本）

- **做法**：用 After Effects / Figma / Keynote 做 **4 状态 × 2 叙事（A/B）** 短片；问卷 + 半结构访谈。
- **能回答**：**叙事 A vs B** 的信任与偏好；不依赖电机与串口稳定性。

### 4.3 建议报告的最小指标（可按 CHI 篇幅裁剪）

- **主观**：NASA-TLX 或简化疲劳量表、对「被监视/被支持」的 Likert。
- **行为**：休息采纳率、通知清除时间、专注段长度分布（注意伦理与样本量说明）。
- **定性**：用设计空间表中的 **含义词** 做编码（用户是否用「伙伴/闹钟/镜子」等隐喻）。

---

## 5. 下一步（仓库内可执行）

- [ ] 在 `gestures.yaml` 或新建 `docs/fluxchi_lelamp_presets_v1.yaml`：**状态 → replay 名 / LED 档**（待硬件后填具体名）。
- [ ] 新建 `docs/fluxchi_elegnt_motion_cards.md`：按 **§2.5.2** 为每状态补全 **Functional / Expressive** 细则，并与预设 YAML 交叉引用。
- [ ] 在 Discord 发帖确认 **官方推荐集成路径**，并更新本文 **§3.1 表格**。
- [ ] 选定论文主叙事 **A 或 B**，在 Introduction 用一段话钉死，避免审稿人认为「社会角色摇摆」。
- [ ] 实验设计：预留 **expression-first vs function-first** 灯侧运动对照（与 ELEGNT 用户研究范式呼应）。

---

## 参考链接

- **ELEGNT**（Apple，非人形灯具机器人运动设计）：[arXiv:2501.12493](https://arxiv.org/abs/2501.12493) · [PDF](https://arxiv.org/pdf/2501.12493)  
- **Elegnt**（Apple ML 研究页，表达性 + 功能性运动）：[machinelearning.apple.com/.../elegnt-expressive-functional-movement](https://machinelearning.apple.com/research/elegnt-expressive-functional-movement)  
- LeLamp 硬件与组装文档：[humancomputerlab/LeLamp](https://github.com/humancomputerlab/LeLamp)  
- 控制运行时：[humancomputerlab/lelamp_runtime](https://github.com/humancomputerlab/lelamp_runtime)  
- 控制教程：[5. LeLamp Control](https://github.com/humancomputerlab/LeLamp/blob/master/docs/5.%20LeLamp%20Control.md)  
- 产品与隐私叙事：[lelamp.com/about](https://www.lelamp.com/about)
