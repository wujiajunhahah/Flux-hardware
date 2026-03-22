# `/ws/vision` 消息格式规格（v1 → v2）

> 状态：**v2 已实现** — `schema: 2` 新增可选 `blendshapes` 字段; `schema: 1` 仍完全兼容。
> 对齐说明：与现有仪表盘 WebSocket `/ws` 一致，**不使用** REST 的 `{ ok, ts, data }` 信封；顶层用 `type` 区分消息种类。

---

## 1. 传输与端点

| 项 | 约定 |
|----|------|
| URL | `ws://{host}:{web_port}/ws/vision`（与 `web/app.py` 同机同端口） |
| 方向 | 主要为**客户端 → 服务端**推送采样帧；服务端可回 `vision_ack` / `error` / `pong` |
| 编码 | UTF-8 JSON 文本帧（与 `/ws` 相同） |
| 建议发送频率 | **10–30 条/秒**；后台标签页降频时允许更低，但需带单调 `seq` |

---

## 2. 客户端 → 服务端

### 2.1 `vision_frame`（主载荷，P0）

每帧一条 JSON。人脸未检出时仍建议发送（便于后端做丢失检测与降权），`face.present` 为 `false`，几何与肤色字段可为 `null` 或省略（实现任选其一，**推荐省略**以减少字节）。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 固定 `"vision_frame"` |
| `schema` | int | 是 | 协议版本：**1**（仅几何）或 **2**（含 BlendShapes） |
| `t` | number | 是 | 客户端时间戳，**秒**，建议 `performance.now()/1000 + 可选时钟锚点` 或 `Date.now()/1000`（实现二选一并文档化，融合层以**窗口内相对顺序**为主） |
| `seq` | int | 是 | 单调递增帧序号，丢包检测用 |
| `tab` | object | 否 | 浏览器上下文，见下 |
| `capture` | object | 否 | 分辨率等，见下 |
| `face` | object | 是 | 检出与质量 |
| `geometry` | object | 否 | 仅当 `face.present === true` 时存在 |
| `blendshapes` | object | 否 | **schema ≥ 2**；疲劳相关 BlendShapes 子集，见下；仅当 `face.present === true` 且模型输出可用时存在 |
| `skin` | object | 否 | 前额/面颊 ROI 均值 RGB，供 rPPG（P1）；P0 可省略 |

**`tab`（可选）**

| 字段 | 类型 | 说明 |
|------|------|------|
| `visible` | boolean | `document.visibilityState === 'visible'` |
| `hostname` | string | 当前页 `location.hostname`，无敏感 path |

**`capture`（可选）**

| 字段 | 类型 | 说明 |
|------|------|------|
| `width` | int | 视频轨宽度 |
| `height` | int | 视频轨高度 |
| `fps_hint` | number | 目标帧率或实际约测 |

**`face`**

| 字段 | 类型 | 说明 |
|------|------|------|
| `present` | boolean | 是否检出人脸 |
| `confidence` | number | 0–1，模型/检测置信度；无则送 `0` |

**`geometry`**（`present=true` 时）

| 字段 | 类型 | 说明 |
|------|------|------|
| `earLeft` | number | 左眼 EAR |
| `earRight` | number | 右眼 EAR |
| `ear` | number | 可选；未送时后端可用 `min(earLeft, earRight)` |
| `mar` | number | 嘴部 MAR |
| `headPoseDeg` | object | `pitch`, `yaw`, `roll`（度）；轴向约定见 §4 |

**`blendshapes`**（schema ≥ 2，可选）

MediaPipe FaceLandmarker 输出的 52 个 BlendShapes 中，仅传输与**疲劳检测**相关的子集，以控制 payload 大小。所有值为 **0–1 浮点**，精度 3 位小数。

| 字段 | 类型 | 说明 |
|------|------|------|
| `eyeBlinkLeft` | number | 左眼闭合程度（替代 EAR，更鲁棒） |
| `eyeBlinkRight` | number | 右眼闭合程度 |
| `eyeSquintLeft` | number | 左眼眯眼（疲劳辅助指标） |
| `eyeSquintRight` | number | 右眼眯眼 |
| `eyeLookDownLeft` | number | 左眼球下看（与困倦相关） |
| `eyeLookDownRight` | number | 右眼球下看 |
| `jawOpen` | number | 张口程度（替代 MAR） |
| `mouthFunnel` | number | 嘴巴圆张（区分哈欠与说话：哈欠时 jawOpen + mouthFunnel 均高，说话时仅 jawOpen 高） |
| `browDownLeft` | number | 左眉下压（皱眉 → 压力/疲劳） |
| `browDownRight` | number | 右眉下压 |
| `browInnerUp` | number | 眉毛上挑（惊讶/挣扎清醒） |
| `cheekPuff` | number | 鼓腮（深呼吸/叹气） |

> **并存策略**：`geometry`（EAR/MAR）与 `blendshapes` 在 schema 2 中**同时发送**。后端优先使用 BlendShapes（精度更高、来自预训练模型），当 `blendshapes` 缺失时自动回退到 `geometry` 的几何计算值。这保证了 schema 1 客户端无需任何修改即可继续工作。

**`skin`**（P1 rPPG）

| 字段 | 类型 | 说明 |
|------|------|------|
| `roiRgb` | object | `r`, `g`, `b` 为 **0–1 浮点**，同一 ROI 上三通道均值 |

**示例（schema 2，人脸可见，含 BlendShapes）**

```json
{
  "type": "vision_frame",
  "schema": 2,
  "t": 1710000000.123,
  "seq": 1842,
  "tab": { "visible": true, "hostname": "github.com" },
  "capture": { "width": 1280, "height": 720, "fps_hint": 30 },
  "face": { "present": true, "confidence": 0.91 },
  "geometry": {
    "earLeft": 0.28,
    "earRight": 0.26,
    "ear": 0.26,
    "mar": 0.12,
    "headPoseDeg": { "pitch": 8.5, "yaw": -12.0, "roll": 1.2 }
  },
  "blendshapes": {
    "eyeBlinkLeft": 0.082,
    "eyeBlinkRight": 0.091,
    "eyeSquintLeft": 0.034,
    "eyeSquintRight": 0.029,
    "eyeLookDownLeft": 0.112,
    "eyeLookDownRight": 0.105,
    "jawOpen": 0.015,
    "mouthFunnel": 0.008,
    "browDownLeft": 0.041,
    "browDownRight": 0.038,
    "browInnerUp": 0.022,
    "cheekPuff": 0.005
  },
  "skin": { "roiRgb": { "r": 0.62, "g": 0.44, "b": 0.37 } }
}
```

**示例（schema 1，人脸可见，无 BlendShapes — 向后兼容）**

```json
{
  "type": "vision_frame",
  "schema": 1,
  "t": 1710000000.123,
  "seq": 1842,
  "tab": { "visible": true, "hostname": "github.com" },
  "capture": { "width": 1280, "height": 720, "fps_hint": 30 },
  "face": { "present": true, "confidence": 0.91 },
  "geometry": {
    "earLeft": 0.28,
    "earRight": 0.26,
    "mar": 0.12,
    "headPoseDeg": { "pitch": 8.5, "yaw": -12.0, "roll": 1.2 }
  },
  "skin": { "roiRgb": { "r": 0.62, "g": 0.44, "b": 0.37 } }
}
```

**示例（无人脸 — schema 无关）**

```json
{
  "type": "vision_frame",
  "schema": 2,
  "t": 1710000000.223,
  "seq": 1843,
  "tab": { "visible": false, "hostname": "github.com" },
  "face": { "present": false, "confidence": 0.0 }
}
```

### 2.2 `ping`（保活）

与 `/ws` 相同：

```json
{ "type": "ping" }
```

服务端应回复：

```json
{ "type": "pong" }
```

---

## 3. 服务端 → 客户端

### 3.1 `vision_ack`（可选，调试用）

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | `"vision_ack"` |
| `lastSeq` | int | 已入队处理的最新 `seq` |
| `dropped` | int | 可选，窗口内丢弃条数 |

### 3.2 `error`

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | `"error"` |
| `code` | string | 如 `"bad_schema"`, `"rate_limit"` |
| `message` | string | 人类可读 |

---

## 4. 头姿轴向约定（须与前端一致）

建议在实现时在 `vision_engine.py` 顶部用注释固定一种约定，例如：

- **pitch**：抬头为正、低头为负（或相反，但前后端必须一致）  
- **yaw / roll**：与 MediaPipe 输出转换后的欧拉角一致  

若首版仅使用「绝对值阈值」，可减少歧义（如只看 `|pitch|`）。

---

## 5. 与 EMG / `latest_payload` 的关系

- EMG 侧 `processing_loop` 约 **每 200ms** 更新 `latest_payload`（`type: "state_update"`），**不是**原始 1 kHz 波形。  
- **融合（fusion_engine）** 应以 **`latest_payload` 中 `stamina` + 时间戳 `timestamp`** 与 vision 侧**按秒对齐的滑动窗口特征**（如 1s PERCLOS、眨眼率）做 `quality_gate + weighted_average`，见产品决策文档。

---

## 6. 后续扩展（不 breaking 时）

- ~~`schema: 2`：可增加 `landmarks_norm`（压缩关键点）、`version` 字符串等。~~ → **已实现**：`schema: 2` 新增疲劳相关 `blendshapes` 子集（12 个系数），见 §2.1。
- `schema: 3`（规划中）：可增加 `landmarks_norm`（压缩关键点）、`au_intensities`（OpenFace AU 强度）、`gaze`（注视方向）等。
- 全量 478 点 **不建议** 在默认通道开启；若调试需要，可用单独查询参数或 `type: "vision_debug"` 限流通道。

---

## 7. 与内部调研文档的交叉引用

- 阈值（EAR、PERCLOS、MAR、pitch）为**可配置调参**，见 [RESEARCH-Camera-Vision.md](./RESEARCH-Camera-Vision.md)。  
- 对外论文引用以 DOI/arXiv 核实为准；本 spec 不涉及文献断言。
