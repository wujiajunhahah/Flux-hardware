# FluxChi iOS

SwiftUI 原生 iOS 客户端，连接 FluxChi EMG Stamina Engine。

## 功能

- **SSE 实时数据流** — 通过 WiFi 连接电脑端 FluxChi 服务器
- **BLE 直连手环** — 使用 CoreBluetooth 直接连接 WAVELETECH 手环（无需 USB 接收器）
- **续航仪表盘** — 实时显示 Stamina 环、三维度、活动分类、EMG 信号
- **Apple HIG** — 遵循 Human Interface Guidelines，支持 Dynamic Type、深色模式

## 架构

```
FluxChi/
├── FluxChiApp.swift              # App 入口 + TabView
├── Models/
│   └── FluxModels.swift          # Codable 数据模型
├── Services/
│   ├── FluxService.swift         # REST API + SSE 客户端
│   └── BLEManager.swift          # CoreBluetooth 直连手环
├── Views/
│   ├── DashboardView.swift       # 主仪表盘
│   ├── StaminaRingView.swift     # 续航环形组件
│   ├── BLEView.swift             # 蓝牙扫描/连接
│   └── SettingsView.swift        # 服务器配置
└── Resources/
    └── Info.plist                # 蓝牙/网络权限
```

## 使用

### 方式 1: WiFi (通过电脑中转)

1. 电脑端启动: `python web/app.py --ble` 或 `--port /dev/tty.usbserial-0001`
2. 手机和电脑连同一 WiFi
3. 在 App 设置页输入电脑 IP 地址和端口 (默认 8000)

### 方式 2: BLE 直连

1. 确保 USB 接收器已拔出
2. 打开 App → 蓝牙 Tab → 扫描
3. 点击 `WL EEG-XXXX` 连接

## 环境要求

- iOS 17.0+
- Xcode 15.0+
- Swift 5.9+

## 在 Xcode 中打开

1. 打开 Xcode → File → New → Project → iOS App
2. 模板选 SwiftUI，产品名 `FluxChi`
3. 将 `ios/FluxChi/` 下所有文件拖入项目
4. 在 Target → Info 中添加 `Info.plist` 中的权限键值
5. 在 Target → Signing & Capabilities 添加 `Background Modes → Uses Bluetooth LE accessories`
6. Build & Run

## BLE 协议

| 参数 | 值 |
|------|-----|
| Service UUID | `974CBE30-3E83-465E-ACDE-6F92FE712134` |
| Data Notify | `974CBE31-3E83-465E-ACDE-6F92FE712134` |
| Write | `974CBE32-3E83-465E-ACDE-6F92FE712134` |
| 帧大小 | 20 字节 |
| EMG 帧 | `0xAA` + seq + 6ch × 24bit |
| IMU 帧 | `0xBB` + seq + 6-axis int16 |
