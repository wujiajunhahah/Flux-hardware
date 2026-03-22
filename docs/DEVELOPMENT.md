# FluxChi 开发指南

目标：**一小时内**在本地跑通 **Web 后端** 或 **iOS（模拟器 + 本机后端）**。假设 macOS，Python 3.10+。

---

## 1. 克隆与 Python 环境

```bash
cd /path/to/harward-gesture
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. 跑 Web 后端 + 浏览器（不依赖 iOS）

```bash
# 无硬件（合成数据，适合联调 UI）
python web/app.py --demo

# USB 串口（路径按本机修改）
python web/app.py --port /dev/tty.usbserial-0001

# 由 Python 通过 BLE 接手环（无 USB 接收器场景）
python web/app.py --ble

# 先启动再在网页里选「数据源」（演示 / USB / BLE），不必写命令行参数
python web/app.py
```

浏览器打开 **`http://127.0.0.1:8000`**（默认端口见终端输出，常见为 `8000`）。

**交互式接口说明**：服务启动后访问 **`http://127.0.0.1:8000/docs`**（Swagger UI）；机器可读契约为 **`http://127.0.0.1:8000/openapi.json`**。与手写 [API.md](./API.md) 不一致时，**以运行中的 JSON 为准**，并欢迎提 PR 更新 `API.md`。

**离线系统自检（无串口/摄像头/HTTP）**：合成 `vision_frame` 与 EMG 窗口，覆盖 VisionEngine / FusionEngine / StaminaEngine / `remove_dc`，可选冒烟 ONNX。

```bash
source .venv/bin/activate
python tools/system_sanity_check.py        # 简要
python tools/system_sanity_check.py -v     # 含 ONNX shape 等提示
```

失败时每条会打印 `FAIL` 与原因，便于 CI 或提交前回归。  
**GitHub**：对 `main` / `master` 的 push 与 PR 会运行同一命令（见仓库根目录 `.github/workflows/ci.yml`）。

---

## 3. 桌面手势训练（与 Web 无关）

```bash
source .venv/bin/activate
python app.py --port /dev/cu.usbserial-0001 --baud 921600 --fs 1000
```

快速复制粘贴版见 [START_HERE.md](../START_HERE.md)。

---

## 4. iOS 应用

1. 安装 **Xcode 15+**（项目目标 iOS 17+，部分能力需更新 SDK）。  
2. 打开 **`ios/FluxChi.xcodeproj`**，选择 Scheme **FluxChi**，运行到模拟器或真机。  
3. **Wi‑Fi 模式联调**：先按 §2 在本机启动 `web/app.py`；模拟器下 App 默认 **`127.0.0.1:8000`**；真机需在设置中填 **电脑的局域网 IP**。  
4. **BLE 直连**：无需 Mac 后端；按 `ios/README.md` 扫描 `WL*` 设备。

**ATS / 明文 HTTP**：若真机无法访问 `http://` 局域网，见 `ios/README.md` 中「Local network HTTP」一节。

**版本号**：上架前同步 **`ios/FluxChi.xcodeproj` 中 `MARKETING_VERSION` / `CURRENT_PROJECT_VERSION`** 与 `FluxMeta.swift` 中 `version`。说明见 [`ios/Distribution/appstore-metadata.md`](../ios/Distribution/appstore-metadata.md)。

---

## 5. 常用路径

| 用途 | 路径 |
|------|------|
| FastAPI 入口 | `web/app.py` |
| 静态仪表盘 | `web/static/` |
| iOS 源码 | `ios/FluxChi/` |
| Widget / Live Activity | `ios/FluxChiLive/` |
| ONNX 与配置 | `model/` |
| 离线管线自检脚本 | `tools/system_sanity_check.py` |
| 接口人类文档 | `docs/API.md` |
| 接口索引与废弃说明 | `docs/API-OVERVIEW.md` |

---

## 6. 安全与密钥

- **勿**将 Discord Webhook、API Key、证书提交到 Git。  
- 使用 CI 时通过 **Secrets** 注入环境变量。

---

## 7. 架构总览

见 [ARCHITECTURE.md](./ARCHITECTURE.md)。
