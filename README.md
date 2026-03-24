# FluxChi

EMG-driven work sustainability engine for digital nomads.

Wristband -> BLE / USB -> ML inference -> Web dashboard + native iOS app.

## What it does

FluxChi reads real-time EMG signals from a WAVELETECH wristband and answers one question: **should you keep working or take a break?**

Instead of a simple fatigue detector, it runs a **StaminaEngine** that accumulates state over time across three dimensions:

| Dimension | What it measures |
|-----------|-----------------|
| Pattern Consistency | How stable your muscle activation patterns remain |
| Baseline Tension | Resting muscle tone -- rises with accumulated strain |
| Sustained Capacity | Long-term degradation of work output quality |

These combine into a **stamina score (0-100)** with four states: `focused` -> `fading` -> `depleted` -> `recovering`.

## Two entry points

This repo contains **two separate Python programs**:

| Entry | Command | Purpose |
|-------|---------|---------|
| **Web API + dashboard** | `python web/app.py --ble` | FastAPI backend (REST + SSE). The iOS app and browser dashboard connect here. |
| **Desktop gesture trainer** | `python app.py --port /dev/cu.usbserial-0001 --baud 921600 --fs 1000` | Standalone serial UI for hand gesture training / data collection. **Not** the web server. |

> See [`START_HERE.md`](START_HERE.md) for the desktop gesture trainer quick-start.

## Quick start (Web API)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# BLE direct (no USB dongle)
python web/app.py --ble

# USB serial
python web/app.py --port /dev/tty.usbserial-0001

# Demo mode (synthetic data)
python web/app.py --demo
```

Open `http://localhost:8000` in your browser.

**Offline regression** (no hardware, no server):

```bash
python tools/system_sanity_check.py
```

## Latest release (Mar 2026)

This release adds a complete "fatigue data flywheel" backend loop:

- **Vision stack integration**
  - OpenFace-compatible per-frame adapter (`AU`, `gaze`, `emotion`)
  - rPPG adapter from ROI RGB (`HR`, `HRV`)
  - Personal vision baseline calibration (`EAR` / `BlendShapes` / head pose)
- **Flywheel APIs**
  - `POST /api/v1/flywheel/label` (`fatigued` / `alert`)
  - `GET /api/v1/flywheel/stats`
  - `POST /api/v1/vision/baseline/start`
  - `GET /api/v1/vision/baseline`
- **Pipeline migration**
  - `web/app.py` processing chain now uses `EmgModule.update()` as the primary path
- **Training pipeline**
  - `scripts/train_flywheel_xgboost.py`
  - Feature export -> XGBoost training -> ONNX export

## Deploy jump page to focux.me (Vercel)

`site/` contains a static "portal page" for `focux.me` that:

- provides one-click links to local FluxChi dashboard/docs
- probes `http://127.0.0.1:8000/api/v1/status` to show online/offline state

Deploy (from repo root):

```bash
npx vercel@latest site --prod
```

Then bind your domain in Vercel:

```bash
npx vercel@latest domains add focux.me
```

> Note: Vercel is used here for the **portal page** only. The hardware gateway (`web/app.py`) still runs on your local/server Python environment.

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design: Web backend, iOS (BLE vs Wi‑Fi), desktop `app.py` |
| [docs/ARCHITECTURE-PRODUCTION-V1.md](docs/ARCHITECTURE-PRODUCTION-V1.md) | Target production architecture for multi-user cloud sync, storage, training, and model release |
| [docs/PLATFORM-CONTRACT-V1.md](docs/PLATFORM-CONTRACT-V1.md) | V1 product API contract, data tables, constraints, indexes, and upload/sync semantics |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Setup, run paths, troubleshooting |
| [docs/API-OVERVIEW.md](docs/API-OVERVIEW.md) | `/api/v1` index, iOS minimal subset, deprecated `/api/*` |
| [docs/API.md](docs/API.md) | Full REST/SSE reference + Swift examples |
| [docs/API-CHANGELOG.md](docs/API-CHANGELOG.md) | API contract changes |
| [CONTRIBUTING.md](CONTRIBUTING.md) | PRs, secrets, scope |

## Architecture

```
                                     BLE 5.0
 +--------------+  <--------------> +-------------------+
 | WAVELETECH   |                   |  iPhone App       |  CoreBluetooth direct
 | EMG wristband|                   |  SwiftUI          |
 | 8ch  1kHz    |     BLE / USB     +-------------------+
 | 6-axis IMU   |  <--------------> |  Python backend   |  FastAPI + uvicorn
 +--------------+                   |  StaminaEngine    |
                                    |  ONNX inference   |
                                    +-------------------+
                                    |  Web dashboard    |  Nothing OS style
                                    +-------------------+
```

## Project structure

```
.
+-- web/
|   +-- app.py                    # FastAPI backend (REST + SSE)
|   +-- static/index.html         # Nothing OS-style dashboard
+-- app.py                        # Desktop gesture trainer (serial UI, NOT web)
+-- src/
|   +-- stream.py                 # SerialEMGStream / BleEMGStream
|   +-- features.py               # 84-dim feature extraction
|   +-- inference.py              # ONNX model inference
|   +-- energy.py                 # StaminaEngine (core model)
|   +-- decision.py               # Recommendation engine
|   +-- ...
+-- ios/                          # Native iOS app (SwiftUI)
|   +-- FluxChi/
|   |   +-- Models/               # SwiftData models (Session, Segment, Snapshot)
|   |   +-- Services/             # FluxService, BLEManager, FluxLogger, etc.
|   |   +-- Views/                # Dashboard, History, Settings, ActiveSession, etc.
|   |   +-- Design/               # FluxTokens, FluxComponents
|   +-- FluxChiLive/              # Widget + Live Activity extension
+-- tools/
|   +-- ble_scan.py               # BLE protocol reverse-engineering tool
+-- docs/
|   +-- ARCHITECTURE.md           # Web + iOS + desktop architecture
|   +-- DEVELOPMENT.md            # Dev setup and runbooks
|   +-- API-OVERVIEW.md           # Endpoint index, iOS subset
|   +-- API-CHANGELOG.md          # Contract changelog
|   +-- API.md                    # REST / SSE API reference (iOS examples)
+-- model/                        # Trained ONNX models
```

## API

The backend exposes a versioned REST API at `/api/v1/`. Index and iOS subset: [`docs/API-OVERVIEW.md`](docs/API-OVERVIEW.md). Full narrative + Swift examples: [`docs/API.md`](docs/API.md). When the server is running, **Swagger UI** is at `/docs` and the machine-readable schema at `/openapi.json`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/state` | GET | Current state (activity, stamina, recommendation) |
| `/api/v1/status` | GET | Server status (model, uptime, sample count) |
| `/api/v1/stream` | GET | SSE real-time event stream |
| `/api/v1/stamina/reset` | POST | Reset stamina engine |
| `/api/v1/stamina/save` | POST | Persist stamina snapshot |

## iOS app

See [`ios/README.md`](ios/README.md) for setup. Two connection modes:

- **WiFi** -- phone polls the Mac backend over REST (500ms interval). Default address: `127.0.0.1:8000` (simulator). For a real device, enter your Mac's LAN IP in Settings.
- **BLE direct** -- CoreBluetooth connects to the wristband without any computer.

## BLE protocol (reverse-engineered)

| Parameter | Value |
|-----------|-------|
| Service UUID | `974CBE30-3E83-465E-ACDE-6F92FE712134` |
| Data Notify | `974CBE31-...` |
| Write | `974CBE32-...` |
| Frame size | 20 bytes |
| EMG frame | `0xAA` + seq + 6ch x 24-bit signed |
| IMU frame | `0xBB` + seq + 6-axis x 16-bit signed |

The wristband **does not advertise its Service UUID** -- scan all devices and filter by name prefix `WL`.

## Tech stack

| Layer | Tech |
|-------|------|
| Hardware | WAVELETECH 8ch EMG + 6-axis IMU, BLE 5.0 |
| Backend | Python, FastAPI, uvicorn, ONNX Runtime |
| ML | scikit-learn -> ONNX, StaminaEngine |
| Web | HTML / CSS / JS, Chart.js, Canvas animations |
| iOS | SwiftUI, SwiftData, CoreBluetooth, Live Activity |
| Tools | xcodegen, bleak, pyserial |

## License

Academic project -- Shenzhen Technology University.
