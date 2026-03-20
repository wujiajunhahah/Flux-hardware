# FluxChi HTTP API 总览

FastAPI 应用在运行时提供：

- **Swagger UI**：`http://<host>:<port>/docs`
- **机器可读契约**：`http://<host>:<port>/openapi.json`

下文为**索引**；字段级说明与 Swift 示例见 [API.md](./API.md)。**若与 `openapi.json` 冲突，以 `openapi.json` 为准**，并应更新 `API.md`。

---

## 1. 版本策略

| 前缀 | 状态 |
|------|------|
| **`/api/v1/*`** | **正式对外**。新客户端必须只依赖此前缀。 |
| **`/api/*`（无 v1）** | **兼容旧客户端**。与 v1 等价别名，**不推荐新代码使用**；未来大版本可能移除。 |

破坏性变更应引入 **`/api/v2/`**（或协商弃用周期），并记入 [API-CHANGELOG.md](./API-CHANGELOG.md)。

---

## 2. 响应信封（REST）

成功：

```json
{ "ok": true, "ts": 1741441200.5, "data": { } }
```

失败：

```json
{ "ok": false, "ts": 1741441200.5, "error": "machine_code", "message": "人类可读说明" }
```

客户端应**先判断 `ok`** 再解析 `data`。

---

## 3. `GET /api/v1/*` 端点索引

| 路径 | 说明 |
|------|------|
| `/api/v1/status` | 服务与模型状态 |
| `/api/v1/model` | 模型元信息 |
| `/api/v1/pulse` | 轻量心跳（适合低频轮询） |
| `/api/v1/stamina` | 续航详情 |
| `/api/v1/decision` | 决策建议 |
| `/api/v1/state` | **完整快照**（iOS 主轮询常用） |
| `/api/v1/emg` | 通道 RMS |
| `/api/v1/timeline` | 提醒历史 |
| `/api/v1/stream` | **SSE** 实时流 |
| `POST /api/v1/stamina/reset` | 重置引擎 |
| `POST /api/v1/stamina/save` | 持久化 |
| `POST /api/v1/stamina/load` | 加载 |
| `GET /api/v1/transport` | 当前数据源模式 + 本机串口列表（网页选 USB） |
| `POST /api/v1/transport/apply` | 切换 `idle` / `demo` / `serial` / `ble`（body 见 [API.md](./API.md)） |

**WebSocket**：`ws://<host>:<port>/ws` — 主要服务 **Web 仪表盘**；iOS 当前以 **REST 轮询** 为主。  
（说明：当前 FastAPI 导出的 `openapi.json` 的 `paths` 里**可能不包含** `/ws`，以 [`web/app.py`](../web/app.py) 中 `@app.websocket` 为准。）

---

## 4. iOS 客户端当前最小依赖（Wi‑Fi 模式）

`FluxService` 等代码实际使用子集如下（实现以 `ios/FluxChi/Services/FluxService.swift` 为准）：

| 用途 | 方法 | 路径 |
|------|------|------|
| 主状态（优先） | **SSE** | `/api/v1/stream`（`event: state` 的 `data:` 行为与 `/api/v1/state` 的 `data` 相同结构的 JSON 快照；失败或断线后**自动回退**为轮询） |
| 主状态（回退） | GET | `/api/v1/state` |
| 状态栏/诊断 | GET | `/api/v1/status` |
| 可选控制 | POST | `/api/v1/stamina/reset`、`/api/v1/stamina/save` |

REST 响应须先判信封 **`ok`**；`ok == false` 时客户端应视为错误（使用 `error` / `message`），勿把 `data` 当有效负载。

新增 App 功能若需更多字段，**优先消费 `/api/v1/state` 已有字段**；不足再提新端点或扩展 `data`，并更新本文件与 `API-CHANGELOG.md`。

---

## 5. JSON 命名约定

- 响应体使用 **snake_case**（与 Python 一致）。  
- Swift 侧通过 `CodingKeys` 映射；文档与示例保持 snake_case，避免混用 camelCase。

---

## 6. 变更记录

见 [API-CHANGELOG.md](./API-CHANGELOG.md)。
