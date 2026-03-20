# FluxChi iOS

SwiftUI native iOS client for the FluxChi EMG Stamina Engine.

## Features

- **REST real-time polling** -- WiFi connection to the Mac backend (500ms interval)
- **BLE direct** -- CoreBluetooth connects to WAVELETECH wristband (no USB dongle or computer needed)
- **Auto mode switch** -- BLE connected = WiFi polling stops; BLE disconnected = auto resume
- **Focus sessions** -- Full-screen immersive focus mode with recording, segmentation, pause/resume
- **Session summary** -- AI-generated NLP summary after each session (iOS 26+ on-device Foundation Models)
- **Dashboard** -- Stamina ring, 3 dimensions, recommendation card, iOS Widget-style insight charts, AI coach chat
- **BIOSORA calendar** -- Dot-matrix data visualization calendar with monthly stats and sparklines
- **History** -- Session list with detail view, charts, export
- **Live Activity + Widget** -- Dynamic Island + Lock Screen + Home Screen widgets (small/medium/large)
- **Onboarding** -- First-launch guide
- **Structured logging** -- FluxLogger with level filtering, export (JSON/text), viewer
- **Performance monitor** -- Launch time, FPS, memory usage, dropped frames
- **Geek data panel** -- Raw EMG signals and RMS visualization
- **Personalization** -- On-device learning from user feedback, calibration offset
- **Data export** -- Sessions as JSON with schema version and metadata
- **Apple HIG** -- Dynamic Type, dark mode, iOS 26 TabBar minimize

## Architecture

```
FluxChi/
+-- FluxChiApp.swift                # App entry + TabView + Focus button + Deep Link + Notifications
+-- Design/
|   +-- FluxTokens.swift            # Design tokens (colors, typography, spacing)
|   +-- FluxComponents.swift        # Reusable UI components
+-- Models/
|   +-- FluxModels.swift            # Codable data models (FluxState, StaminaData, etc.)
|   +-- Session.swift               # SwiftData models (Session, Segment, Snapshot, Feedback)
+-- Services/
|   +-- FluxService.swift           # REST API polling client
|   +-- BLEManager.swift            # CoreBluetooth direct wristband connection
|   +-- SessionManager.swift        # Recording session lifecycle
|   +-- FluxLogger.swift            # Structured logging system
|   +-- ExportManager.swift         # Session data export
|   +-- PerformanceMonitor.swift    # Launch time, FPS, memory tracking
|   +-- PersonalizationManager.swift # On-device learning
|   +-- OnDeviceStaminaEngine.swift # Local stamina calculation
|   +-- EMGFeatureExtractor.swift   # Feature extraction from raw EMG
|   +-- NLPSummaryEngine.swift      # AI session summary (Foundation Models)
|   +-- BodyInsightEngine.swift     # Daily insight generation
|   +-- SummaryEngine.swift         # Rule-based session summary
|   +-- AlertManager.swift          # Break reminders
|   +-- LiveActivityManager.swift   # Dynamic Island + Lock Screen
|   +-- WidgetDataManager.swift     # Widget data bridge
|   +-- ModelContext+SaveLogging.swift # SwiftData save with error logging
+-- Views/
|   +-- DashboardView.swift         # Main dashboard with insights + charts
|   +-- ActiveSessionView.swift     # Full-screen focus session
|   +-- SessionSummarySheet.swift   # Post-session AI summary
|   +-- HistoryView.swift           # Session history list
|   +-- CalendarView.swift          # BIOSORA dot-matrix calendar
|   +-- SettingsView.swift          # Settings (BLE, server, logs, perf, geek data)
|   +-- OnboardingView.swift        # First-launch guide
|   +-- ConnectionGuideSheet.swift  # Connection help
|   +-- FeedbackView.swift          # Post-session feedback
|   +-- RecorderView.swift          # Recording controls
|   +-- SessionDetailView.swift     # Single session detail
|   +-- SessionChartsView.swift     # Session data charts
|   +-- StaminaRingView.swift       # Stamina ring component
|   +-- GeekDataPanel.swift         # Raw EMG data viewer
|   +-- LogViewerView.swift         # Log browser
+-- ML/
|   +-- ActivityClassifier.mlpackage # CoreML activity model
+-- Resources/
    +-- Info.plist                   # Permissions (BLE, local network)
    +-- PrivacyInfo.xcprivacy        # Privacy manifest

FluxChiLive/                        # Widget + Live Activity extension
+-- FluxChiWidgets.swift            # Small/Medium/Large widgets
+-- FluxChiLiveActivity.swift       # Dynamic Island
+-- WidgetDataReader.swift          # Shared data reader
```

## Usage

### Mode 1: WiFi (via Mac backend)

1. Start backend on Mac: `python web/app.py --ble` or `--port /dev/tty.usbserial-0001`
2. Connect phone and Mac to the same WiFi
3. In app Settings, enter your Mac's IP and port (default 8000)

> **Simulator / local**: default address is `127.0.0.1:8000` -- works out of the box.
>
> **Real device**: you must change the address to your Mac's LAN IP (e.g. `192.168.1.x`). `127.0.0.1` on a real device points to the phone itself.

### Mode 2: BLE direct

1. Unplug USB dongle (wristband can only pair with one host)
2. Open app -> Settings -> Bluetooth -> Scan
3. Tap `WL EEG-XXXX` to connect

## Local network HTTP (ATS)

The app connects to the backend over **plain HTTP** (not HTTPS). iOS App Transport Security may block this on some configurations.

Current status: `Info.plist` includes `NSLocalNetworkUsageDescription`. If you encounter connection failures on a real device:

1. Add to `Info.plist`:
   ```xml
   <key>NSAppTransportSecurity</key>
   <dict>
       <key>NSAllowsLocalNetworking</key>
       <true/>
   </dict>
   ```
2. Or switch the backend to HTTPS with a self-signed cert.

## Requirements

- iOS 17.0+ (iOS 26+ for Foundation Models NLP summary)
- Xcode 16.0+ (Xcode 26 beta for iOS 26 features)
- Swift 5.9+

## Build

Open `ios/FluxChi.xcodeproj` in Xcode, select your target device, and Build & Run.

## BLE protocol

| Parameter | Value |
|-----------|-------|
| Service UUID | `974CBE30-3E83-465E-ACDE-6F92FE712134` |
| Data Notify | `974CBE31-3E83-465E-ACDE-6F92FE712134` |
| Write | `974CBE32-3E83-465E-ACDE-6F92FE712134` |
| Frame size | 20 bytes |
| EMG frame | `0xAA` + seq + 6ch x 24bit |
| IMU frame | `0xBB` + seq + 6-axis int16 |
