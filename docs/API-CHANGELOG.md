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
