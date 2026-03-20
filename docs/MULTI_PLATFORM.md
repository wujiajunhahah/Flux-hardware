# FluxChi 多端对齐说明

原则：**判断、规范化、洞察、归档在 Python 网关（后端）完成**；**手机与网页尽量只做展示与最少操作**（开始/结束、同步、下载）。

## 信号路径

| 端 | 物理通道 | 说明 |
|----|-----------|------|
| 电脑 + USB 串口 | 8 | `SerialEMGStream`，与 `CHANNEL_COUNT=8` 一致 |
| 手机 + BLE | 6 | `BLEManager`，帧内 6 路 RMS；**不在前端补通道** |

## 规范层（Canonical）

- 模块：[`src/canonical_signal.py`](../src/canonical_signal.py)
- **分析用 RMS 向量固定长度 8**：不足补 `0`，超出截断；`signalMeta` 中保留 `activeChannels` / `note` 说明真实有效路数。
- 所有进入归档的 JSON（网页录制、手机 `ExportPackage` 经 `ingest`）都会经过 **`canonicalize_export_package`**。

## 会话与互通

- **落盘目录**：`data/sessions/`（已在 `.gitignore` 的 `data/` 下）。
- **归档文件名（与 iOS 分享/导出一致）**：`fluxchi_{yyyyMMdd_HHmmss}_{session_uuid}.json`（`session_uuid` 为全小写 UUID，时间戳为**保存到网关时**的墙钟；同一 `session.id` 再次 `ingest` 会删掉旧 `fluxchi_*_{id}.json` 与旧式 `{id}.json` 后重写）。JSON 根级字段 **`suggestedArchiveFileName`** 与磁盘文件名一致；`GET /api/v1/sessions` 每条含 **`archiveFile`**（实际文件名）与 **`suggestedArchiveFileName`**。
- **iOS**：`ExportManager.export` 写入 **`suggestedArchiveFileName`**，分享临时文件与 AirDrop 文件名同为上述格式（时间为**导出时刻**墙钟；上传网关后服务端可能用**ingest 时刻**重新命名）。
- **网页**：`POST /api/v1/sessions/web/start` → `stop` → 返回 `insight` + `download_url`；快照由 `processing_loop` 每 **500ms** 采一次（与 iOS `snapshotIntervalMs` 一致）。
- **手机**：专注结束页 **「同步到电脑」** → `POST /api/v1/sessions/ingest`，Body 与 `ExportManager.export` JSON **同构**。
- **列表与下载**：`GET /api/v1/sessions`、`GET /api/v1/sessions/{id}`、`GET /api/v1/sessions/{id}/file`。
- **洞察文案**：[`src/session_insight.py`](../src/session_insight.py)（确定性规则，后续可换模型，接口不变）。

## 手机网关地址

使用与 Wi‑Fi 模式相同的 UserDefaults：`flux_host`、`flux_port`（默认端口 **8000**）。请填 **运行 `web/app.py` 的电脑局域网 IP**，不要用 `127.0.0.1`（指手机自身）。

## 分析脚本建议

- 只读 `snapshots[].rms`（长度 8）+ `signalMeta.canonicalEmgChannels` + `activeChannels` / `note` 判断哪些通道可信。
