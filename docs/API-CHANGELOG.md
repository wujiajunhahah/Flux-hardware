# FluxChi API 变更日志

记录 **`/api/v1/*` 对外契约**的破坏性或与客户端相关的变更。内部重构（不改变 JSON 形状）可不记。

格式建议：

```text
## YYYY-MM-DD — 标题
### Breaking
- ...
### Added
- ...
### Fixed
- ...
```

---

## 基准线（文档建立时）

- **正式前缀**：`/api/v1/`。  
- **兼容前缀**：无版本 `/api/*` 与 v1 并行，新代码勿用。  
- **信封**：`ok` / `ts` / `data` 或 `error` + `message`。  
- **iOS Wi‑Fi 最小集**：`GET .../state`、`GET .../status`；`POST .../stamina/reset|save`。

---

## 历史条目

### 2026-03-21 — CI、离线自检、视觉告警单一来源（无 JSON 契约变更）

- **仓库**：GitHub Actions（`.github/workflows/ci.yml`）在 **push / PR** 至 `main` 或 `master` 时执行 `pip install -r requirements.txt` 与 **`python tools/system_sanity_check.py`**（`model/` 未提交时 ONNX 段自动跳过，仍可通过）。
- **开发者**：见 [DEVELOPMENT.md](./DEVELOPMENT.md) §2「离线系统自检」；贡献流程见 [CONTRIBUTING.md](../CONTRIBUTING.md)。
- **内部实现**：`VisionReading` → 告警 ID 列表由 **`src/vision_engine.vision_reading_alerts`** 统一生成；**不改变** `state_update` / `vision` / `fusion` 对外字段形状。

### 2026-03-20 — `fusion.stamina` 可空、`vision.stale`、眨眼统计修复

- **Breaking（轻微）**：当 EMG 与 Vision 均不可用于融合时，`state_update` 内 **`fusion.stamina`** 为 **`null`**（不再使用占位 `50`）；`fusion.state` 仍为 **`unknown`**。客户端应用 `source === "none"` 或 `stamina == null` 时勿把主环当作有效 0–100 分。
- **新增**：`state_update` 中若含 **`vision`**，增加布尔字段 **`vision.stale`**：服务端在约 **2.5s** 内未收到新 `vision_frame` 时为 `true`；**过期视觉不参与 `FusionEngine` 计算**（与仍附带的快照解耦，便于调试末帧）。
- **Fixed**：`VisionEngine` 眨眼状态机在睁眼帧登记完整眨眼，**`blink_rate` / `blink_count`** 与 PERCLOS 一致可增长。
- **Web 仪表盘**：无手环且无有效综合分时主环显示 **「—」**（不再回退为假 **100**）；融合卡片文案与过期提示对齐上述语义。

### 2026-03-20 — WebSocket `/ws/vision` 与实时 `fusion` / `vision` 字段

- **新增**：WebSocket 端点 `ws://{host}:{port}/ws/vision`，浏览器按 [VISION-WEBSOCKET-SPEC.md](./VISION-WEBSOCKET-SPEC.md) 发送 `vision_frame` JSON（不传视频帧）。
- **实时推送**：`state_update`（`/ws`、`/api/v1/stream` 等与现有仪表盘一致）在存在视觉流时增加可选字段 **`vision`**（`VisionReading.to_dict()`）、**`fusion`**（`FusionEngine` 输出摘要）。无 EMG 数据源时仍会约每 200ms 广播仅含 `vision`/`fusion` 的空 EMG 占位帧，便于「只开摄像头」调试。
- **静态页**：`/static/vision.html` + `/static/vision.js`（MediaPipe Tasks CDN）。
- **Web 主仪表盘**：主续航环与趋势图优先显示 **`fusion.stamina`**；新增「综合续航」卡片（来源、权重、视觉摘要、告警标签）；窄屏样式优化（`max-width: 420px`）。
- **`GET /api/v1/transport`** / **`GET /api/v1/status`**：增加 **`vision`**（`ws_clients`、`receiving`、`last_frame_age_sec`）与 **`sources_note`**；仪表盘分「手环 EMG」「摄像头」两行，二者可同时启用。
- **摄像头 UX**：抽取 `web/static/vision-capture.js`；主仪表盘支持**本页**开启摄像头（先 `getUserMedia` 再加载模型与 WS，减少「授权后无反馈」）；`vision.html` 保留为独立调试页并含返回仪表盘链接。
- **BlendShapes（摄入）**：`vision_frame` 可使用 **`schema: 2`** 并附带疲劳相关 **`blendshapes`** 系数（见 [VISION-WEBSOCKET-SPEC.md](./VISION-WEBSOCKET-SPEC.md) §2.1）。服务端 **`VisionEngine`** 对眨眼/PERCLOS 与哈欠**优先**使用该字段，缺失时回退 **`geometry`** 的 EAR/MAR；`state_update` 里的 **`vision`** 仍为融合后的 **`VisionReading`** 快照（不回传原始 blendshapes）。

### 2026-03-20 — 会话归档文件名与时间序列

- **落盘**：`data/sessions/fluxchi_yyyyMMdd_HHmmss_{uuid}.json`（小写 UUID）；同 id 再保存时替换旧文件。
- **JSON**：根级可选/推荐字段 **`suggestedArchiveFileName`**（与落盘名一致）。
- **`GET /api/v1/sessions`**：列表项增加 **`archiveFile`**、**`suggestedArchiveFileName`**。
- **`GET .../sessions/{id}/file`**：`Content-Disposition` 文件名与磁盘 **`path.name`** 一致（不再使用 `fluxchi_{id}.json` 简名）。
- **iOS**：`ExportPackage` 含 **`suggestedArchiveFileName`**；系统分享用同一时间戳 + 全 UUID 文件名。

### 2026-03-20 — iOS Wi‑Fi：`FluxService`

- **客户端**：优先连接 SSE `GET /api/v1/stream`，断线或失败后回退为 `GET /api/v1/state` 轮询（服务端契约未变）。
- **客户端**：REST 信封 `ok == false` 时按错误处理，并更新连接/错误提示。

### 2026-03-20 — Web：运行时切换数据源

- **新增**：`GET /api/v1/transport`、`POST /api/v1/transport/apply`；可无 `--demo/--port/--ble` 启动服务，在网页「数据源」选择演示 / USB 串口 / BLE / 断开。
- **`/api/v1/status`**：`data` 增加可选字段 `transport_mode`、`serial_port`、`ble_address`（旧客户端应忽略未知字段）。
- **`POST .../transport/apply`（serial）**：`port` 可省略；本机**仅检测到 1 个串口时自动连接**；多串口时需下拉或手动路径。成功时 `data` 可含 `auto_selected`。
- **`GET .../transport`**：`serial_ports` 按 USB 桥优先排序并带 `likely_emg_bridge`；`?include_all=1` 附加 `serial_ports_other`；有数据流时返回 `stream_stats`（`emg_frames` 等）。macOS 侧打开串口时自动将 `/dev/tty.*` 规范为 `/dev/cu.*`。
- **会话归档与互通**：`POST .../sessions/web/start|stop`、`POST .../sessions/ingest`、`GET .../sessions`、`GET .../sessions/{id}`、`.../file`、`.../insight`。归档落盘 `data/sessions/`；`ingest` 与 iOS `ExportPackage` 同构并由后端 **canonical 8 路 RMS** + `session_insight` 写 `summary.text`（若原为空）。详见 [MULTI_PLATFORM.md](./MULTI_PLATFORM.md)。

（在此追加；与 Git tag / App 版本无强制对应，但大版本发布时建议补一条汇总。）
