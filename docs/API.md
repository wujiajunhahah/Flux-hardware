# FluxChi API Reference

> **维护说明**：端点索引、iOS 最小子集、无版本 `/api/*` 废弃策略见 [API-OVERVIEW.md](./API-OVERVIEW.md)。契约变更见 [API-CHANGELOG.md](./API-CHANGELOG.md)。服务运行时的权威机器可读描述为 **`/openapi.json`**，交互文档为 **`/docs`**（Swagger UI）；若与此 Markdown 不一致，以运行实例为准并欢迎修正本文。

## Overview

FluxChi exposes a versioned REST API, an SSE stream, and a WebSocket endpoint. All three deliver the same core data — choose the transport that fits your client.

| Transport | Endpoint | Best for |
|-----------|----------|----------|
| REST | `GET /api/v1/*` | iOS polling, one-shot queries |
| SSE | `GET /api/v1/stream` | iOS `URLSession`, real-time with auto-reconnect |
| WebSocket | `ws://host/ws` | Web dashboard, lowest latency |

Base URL: `http://<host>:8000`

Interactive docs (live when server is running):
- Swagger UI — `/docs`
- ReDoc — `/redoc`

---

## Quick Start

```bash
# Real hardware
python web/app.py --port /dev/tty.usbserial-0001

# No hardware (demo)
python web/app.py --demo

# 无初始数据源：启动后在网页「数据源」里选 USB / 演示 / BLE
python web/app.py --web-port 8000

# Defense demo (10x speed)
python web/app.py --demo --speed 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | — | Serial port path |
| `--demo` | false | Synthetic EMG data |
| `--host` | `0.0.0.0` | Bind address |
| `--web-port` | `8000` | HTTP port |
| `--model-dir` | `model` | ONNX model directory |
| `--speed` | `1.0` | Stamina drain/recovery multiplier |

---

## Response Envelope

Every response follows a consistent structure:

```json
{
  "ok": true,
  "ts": 1741441200.5,
  "data": { ... }
}
```

On error:

```json
{
  "ok": false,
  "ts": 1741441200.5,
  "error": "no_data",
  "message": "传感器数据尚未就绪"
}
```

Always check `ok` before reading `data`.

---

## Endpoints

### `GET /api/v1/transport` — Data source + serial ports

用于网页仪表盘填充「USB 串口」下拉列表，并显示当前模式。

查询参数：`include_all=true` 时额外返回 `serial_ports_other`（耳机/蓝牙虚拟串口等低优先级项）。

`serial_ports` 已按启发式**排序并筛选**：优先 USB 串口桥（CP2102、CH340、FTDI 等），条目可含 `likely_emg_bridge`。连接 `serial` 后 `stream_stats` 会反映是否真正收到协议帧（`emg_frames` 长期为 0 多为选错端口或手环未发数）。

```json
{
  "ok": true,
  "ts": 1741441200.5,
  "data": {
    "mode": "serial",
    "demo_mode": false,
    "serial_port": "/dev/cu.usbserial-0001",
    "ble_address": null,
    "serial_ports": [
      {"device": "/dev/cu.usbserial-0001", "description": "CP2102 …", "likely_emg_bridge": true}
    ],
    "serial_ports_other": [],
    "stream_stats": {"emg_frames": 1204, "total_frames": 1300, "imu_frames": 0, "dropped_frames": 0},
    "signal_hint": "…"
  }
}
```

`mode` 取值：`idle`（无流）、`demo`、`serial`、`ble`。

---

### `POST /api/v1/transport/apply` — Switch data source

运行时切换数据源，等价于用不同参数重启服务（无需重启进程）。

**Body（JSON）**

| 字段 | 说明 |
|------|------|
| `mode` | 必填：`idle` \| `demo` \| `serial` \| `ble` |
| `port` | `serial` 时**可选**：写明路径则用之；**省略且本机仅检测到 1 个串口时由服务端自动选用**；0 个或多个串口时必须指定 |
| `address` | `ble` 时可选；省略或空字符串则自动扫描手环 |

示例：

```json
{"mode": "serial", "port": "/dev/tty.usbserial-1420"}
```

```json
{"mode": "demo"}
```

```json
{"mode": "idle"}
```

成功时 `data` 含 `message` 等人读说明；失败时 `ok: false`，常见 `error`：`bad_request`、`open_failed`。

---

### `GET /api/v1/pulse` — Lightweight Heartbeat

Minimal payload for iOS background polling. ~200 bytes.

```json
{
  "ok": true,
  "ts": 1741441200.5,
  "data": {
    "stamina": 72.5,
    "state": "focused",
    "recommendation": "keep_working",
    "urgency": 0.0,
    "suggested_work_min": 26,
    "suggested_break_min": 0
  }
}
```

---

### `GET /api/v1/stamina` — Full Stamina Details

```json
{
  "ok": true,
  "ts": 1741441200.5,
  "data": {
    "stamina": 72.5,
    "state": "focused",
    "dimensions": {
      "consistency": 0.85,
      "tension": 0.12,
      "fatigue": 0.08
    },
    "rates": {
      "drain_per_min": 1.65,
      "recovery_per_min": 4.50
    },
    "suggestion": {
      "work_min": 26,
      "break_min": 0
    },
    "session": {
      "continuous_work_min": 18.5,
      "total_work_min": 45.2
    }
  }
}
```

**Dimensions:**

| Field | Range | Meaning |
|-------|-------|---------|
| `consistency` | 0–1 | Pattern stability → focus proxy |
| `tension` | 0–1 | Residual muscle tension → stress |
| `fatigue` | 0–1 | MDF spectral decline → endurance |

**States:**

| State | Stamina | Icon |
|-------|---------|------|
| `focused` | > 60 | 🔥 |
| `fading` | 30–60 | ⚡ |
| `depleted` | < 30 | 🔋 |
| `recovering` | resting | 🌿 |

---

### `GET /api/v1/decision` — Recommendation

```json
{
  "ok": true,
  "ts": 1741441200.5,
  "data": {
    "state": "fading",
    "recommendation": "take_break",
    "urgency": 0.55,
    "reasons": ["效率在下降（续航 45%），建议 8 分钟内休息"],
    "stamina": 45.0,
    "suggested_work_min": 8,
    "suggested_break_min": 5
  }
}
```

| Recommendation | When |
|----------------|------|
| `keep_working` | Stamina healthy, urgency < 0.5 |
| `take_break` | Urgency ≥ 0.5 |
| `start_working` | Resting + stamina ≥ 50 |
| `rest_more` | Resting + stamina < 50 |

---

### `GET /api/v1/state` — Full Snapshot

Returns the complete payload (identical to SSE/WebSocket frames). Use this for one-shot queries when you need everything.

---

### `GET /api/v1/emg` — Raw EMG RMS

```json
{
  "ok": true,
  "ts": 1741441200.5,
  "data": {
    "channels": 8,
    "rms": [42.1, 38.5, 15.2, 12.8, 8.3, 6.1, 5.5, 4.2],
    "sample_count": 5000
  }
}
```

---

### `GET /api/v1/status` — Server Info

```json
{
  "ok": true,
  "ts": 1741441200.5,
  "data": {
    "connected": true,
    "model_loaded": true,
    "demo_mode": false,
    "speed": 1.0,
    "sample_rate": 1000,
    "channels": 8,
    "uptime_sec": 342.5,
    "websocket_clients": 1
  }
}
```

---

### `GET /api/v1/model` — Model Info

### `GET /api/v1/timeline?limit=50` — Alert History

### `POST /api/v1/stamina/reset` — Reset to 100%

### `POST /api/v1/stamina/save` — Persist to Disk

### `POST /api/v1/stamina/load` — Restore from Disk

---

## SSE Stream

`GET /api/v1/stream`

Server-Sent Events — the recommended transport for iOS. Native `URLSession` support, automatic reconnection, no third-party dependencies.

Each frame:

```
event: state
data: {"type":"state_update","timestamp":...,"stamina":{...},"decision":{...}}
```

### Swift — EventSource with URLSession

```swift
import Foundation

class FluxChiSSE {
    private var task: URLSessionDataTask?
    var onUpdate: ((FluxState) -> Void)?

    func connect(host: String = "localhost", port: Int = 8000) {
        let url = URL(string: "http://\(host):\(port)/api/v1/stream")!
        var request = URLRequest(url: url)
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.timeoutInterval = .infinity

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = .infinity
        config.timeoutIntervalForResource = .infinity

        let session = URLSession(configuration: config, delegate: SSEDelegate(onUpdate: { [weak self] state in
            self?.onUpdate?(state)
        }), delegateQueue: nil)

        task = session.dataTask(with: request)
        task?.resume()
    }

    func disconnect() {
        task?.cancel()
    }
}

class SSEDelegate: NSObject, URLSessionDataDelegate {
    let onUpdate: (FluxState) -> Void
    private var buffer = Data()

    init(onUpdate: @escaping (FluxState) -> Void) {
        self.onUpdate = onUpdate
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        buffer.append(data)
        guard let text = String(data: buffer, encoding: .utf8) else { return }

        let lines = text.components(separatedBy: "\n\n")
        for block in lines.dropLast() {
            if let dataLine = block.components(separatedBy: "\n")
                .first(where: { $0.hasPrefix("data: ") }) {
                let json = String(dataLine.dropFirst(6))
                if let jsonData = json.data(using: .utf8),
                   let state = try? JSONDecoder().decode(FluxState.self, from: jsonData) {
                    DispatchQueue.main.async { self.onUpdate(state) }
                }
            }
        }
        if let last = lines.last {
            buffer = last.data(using: .utf8) ?? Data()
        }
    }
}
```

---

## iOS Integration Guide

### Data Models (Codable)

```swift
struct FluxResponse<T: Codable>: Codable {
    let ok: Bool
    let ts: Double
    let data: T?
    let error: String?
    let message: String?
}

struct FluxState: Codable {
    let type: String
    let timestamp: Double
    let activity: String
    let confidence: Double
    let probabilities: [String: Double]?
    let rms: [Double]?
    let stamina: StaminaData?
    let decision: DecisionData?
}

struct StaminaData: Codable {
    let value: Double
    let state: String
    let consistency: Double
    let tension: Double
    let fatigue: Double
    let drainRate: Double
    let recoveryRate: Double
    let suggestedWorkMin: Double
    let suggestedBreakMin: Double
    let continuousWorkMin: Double
    let totalWorkMin: Double

    enum CodingKeys: String, CodingKey {
        case value, state, consistency, tension, fatigue
        case drainRate = "drain_rate"
        case recoveryRate = "recovery_rate"
        case suggestedWorkMin = "suggested_work_min"
        case suggestedBreakMin = "suggested_break_min"
        case continuousWorkMin = "continuous_work_min"
        case totalWorkMin = "total_work_min"
    }
}

struct DecisionData: Codable {
    let state: String
    let recommendation: String
    let urgency: Double
    let reasons: [String]
    let stamina: Double
    let suggestedWorkMin: Double
    let suggestedBreakMin: Double

    enum CodingKeys: String, CodingKey {
        case state, recommendation, urgency, reasons, stamina
        case suggestedWorkMin = "suggested_work_min"
        case suggestedBreakMin = "suggested_break_min"
    }
}
```

### API Client

```swift
actor FluxChiClient {
    let baseURL: URL

    init(host: String = "localhost", port: Int = 8000) {
        self.baseURL = URL(string: "http://\(host):\(port)")!
    }

    func pulse() async throws -> PulseData {
        try await get("/api/v1/pulse")
    }

    func stamina() async throws -> StaminaDetail {
        try await get("/api/v1/stamina")
    }

    func resetStamina() async throws {
        let _: EmptyOK = try await post("/api/v1/stamina/reset")
    }

    private func get<T: Codable>(_ path: String) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        let (data, _) = try await URLSession.shared.data(from: url)
        let response = try JSONDecoder().decode(FluxResponse<T>.self, from: data)
        guard response.ok, let result = response.data else {
            throw FluxError.apiError(response.error ?? "unknown")
        }
        return result
    }

    private func post<T: Codable>(_ path: String) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        let (data, _) = try await URLSession.shared.data(for: request)
        let response = try JSONDecoder().decode(FluxResponse<T>.self, from: data)
        guard response.ok, let result = response.data else {
            throw FluxError.apiError(response.error ?? "unknown")
        }
        return result
    }
}

enum FluxError: Error {
    case apiError(String)
}

struct PulseData: Codable {
    let stamina: Double
    let state: String
    let recommendation: String
    let urgency: Double
    let suggestedWorkMin: Double
    let suggestedBreakMin: Double

    enum CodingKeys: String, CodingKey {
        case stamina, state, recommendation, urgency
        case suggestedWorkMin = "suggested_work_min"
        case suggestedBreakMin = "suggested_break_min"
    }
}

struct StaminaDetail: Codable {
    let stamina: Double
    let state: String
    let dimensions: Dimensions
    let rates: Rates
    let suggestion: Suggestion
    let session: Session

    struct Dimensions: Codable {
        let consistency: Double
        let tension: Double
        let fatigue: Double
    }
    struct Rates: Codable {
        let drainPerMin: Double
        let recoveryPerMin: Double
        enum CodingKeys: String, CodingKey {
            case drainPerMin = "drain_per_min"
            case recoveryPerMin = "recovery_per_min"
        }
    }
    struct Suggestion: Codable {
        let workMin: Double
        let breakMin: Double
        enum CodingKeys: String, CodingKey {
            case workMin = "work_min"
            case breakMin = "break_min"
        }
    }
    struct Session: Codable {
        let continuousWorkMin: Double
        let totalWorkMin: Double
        enum CodingKeys: String, CodingKey {
            case continuousWorkMin = "continuous_work_min"
            case totalWorkMin = "total_work_min"
        }
    }
}

struct EmptyOK: Codable {
    let stamina: Double?
    let loaded: Bool?
}
```

### SwiftUI View Example

```swift
import SwiftUI

struct StaminaView: View {
    @StateObject private var vm = StaminaViewModel()

    var body: some View {
        VStack(spacing: 24) {
            ZStack {
                Circle()
                    .stroke(Color(.systemGray5), lineWidth: 16)
                Circle()
                    .trim(from: 0, to: vm.stamina / 100)
                    .stroke(vm.staminaColor, style: StrokeStyle(lineWidth: 16, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 0.5), value: vm.stamina)
                VStack(spacing: 4) {
                    Text("\(Int(vm.stamina))%")
                        .font(.system(size: 48, weight: .bold, design: .rounded))
                    Text(vm.stateLabel)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(width: 200, height: 200)

            HStack(spacing: 32) {
                DimensionPill(title: "专注度", value: vm.consistency, color: .blue)
                DimensionPill(title: "紧张度", value: vm.tension, color: .orange)
                DimensionPill(title: "退化", value: vm.fatigue, color: .red)
            }

            if !vm.recommendation.isEmpty {
                Text(vm.recommendation)
                    .font(.body)
                    .padding()
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
        }
        .padding()
        .task { await vm.startPolling() }
    }
}

struct DimensionPill: View {
    let title: String
    let value: Double
    let color: Color

    var body: some View {
        VStack(spacing: 4) {
            Text("\(Int(value * 100))%")
                .font(.system(.title3, design: .rounded).bold())
                .foregroundStyle(color)
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}

@MainActor
class StaminaViewModel: ObservableObject {
    @Published var stamina: Double = 100
    @Published var stateLabel = "—"
    @Published var consistency: Double = 0
    @Published var tension: Double = 0
    @Published var fatigue: Double = 0
    @Published var recommendation = ""

    private let client = FluxChiClient()

    var staminaColor: Color {
        stamina > 60 ? .green : stamina > 30 ? .yellow : .red
    }

    func startPolling() async {
        while !Task.isCancelled {
            do {
                let data = try await client.stamina()
                stamina = data.stamina
                stateLabel = data.state
                consistency = data.dimensions.consistency
                tension = data.dimensions.tension
                fatigue = data.dimensions.fatigue
                recommendation = data.suggestion.workMin > 0
                    ? "可专注 \(Int(data.suggestion.workMin)) 分钟"
                    : "建议休息 \(Int(data.suggestion.breakMin)) 分钟"
            } catch {
                stateLabel = "连接中..."
            }
            try? await Task.sleep(for: .seconds(5))
        }
    }
}
```

### iOS Background Polling

```swift
import BackgroundTasks

func scheduleStaminaCheck() {
    let request = BGAppRefreshTaskRequest(identifier: "com.fluxchi.staminaCheck")
    request.earliestBeginDate = Date(timeIntervalSinceNow: 30)
    try? BGTaskScheduler.shared.submit(request)
}

func handleStaminaCheck(task: BGAppRefreshTask) {
    let client = FluxChiClient()
    Task {
        defer { task.setTaskCompleted(success: true) }
        guard let pulse = try? await client.pulse() else { return }
        if pulse.urgency >= 0.5 {
            sendLocalNotification(
                title: "FluxChi",
                body: "续航 \(Int(pulse.stamina))%，建议休息 \(Int(pulse.suggestedBreakMin)) 分钟"
            )
        }
        scheduleStaminaCheck()
    }
}
```

---

## Polling Strategy

| Context | Transport | Endpoint | Interval |
|---------|-----------|----------|----------|
| Web dashboard | WebSocket | `/ws` | Real-time (~5 fps) |
| iOS foreground | SSE | `/api/v1/stream` | Real-time |
| iOS background | REST | `/api/v1/pulse` | 30–60s |
| Notification check | REST | `/api/v1/pulse` | On trigger |
| Debug / one-shot | REST | `/api/v1/state` | On demand |

---

## Architecture

```
┌──────────────┐     Serial/BLE      ┌──────────────┐
│  WAVELETECH  │ ──────────────────▶ │  FluxChi API │
│   Wristband  │     1000 Hz EMG     │   (FastAPI)  │
└──────────────┘                     └──────┬───────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
             ┌──────▼──────┐        ┌───────▼──────┐       ┌───────▼──────┐
             │   REST API  │        │  SSE Stream  │       │  WebSocket   │
             │  /api/v1/*  │        │ /api/v1/     │       │    /ws       │
             │             │        │    stream    │       │              │
             └──────┬──────┘        └───────┬──────┘       └───────┬──────┘
                    │                       │                       │
             ┌──────▼──────┐        ┌───────▼──────┐       ┌───────▼──────┐
             │   iOS App   │        │   iOS App    │       │     Web      │
             │ (bg poll)   │        │ (foreground) │       │  Dashboard   │
             └─────────────┘        └──────────────┘       └──────────────┘
```
