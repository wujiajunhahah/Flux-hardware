# 文档说明

## 核心（主线：Web + iOS + 桌面工具）

| 文档 | 说明 |
|------|------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 系统架构：三条运行时、iOS 与 Web 的并行/耦合关系 |
| [ARCHITECTURE-PRODUCTION-V1.md](./ARCHITECTURE-PRODUCTION-V1.md) | 面向真实产品与多用户演进的目标架构、服务边界、数据边界与路线图 |
| [PLATFORM-CONTRACT-V1.md](./PLATFORM-CONTRACT-V1.md) | 产品化 V1 的 API Contract、数据表、约束与索引，供后端/iOS/算法统一对齐 |
| [DEVELOPMENT.md](./DEVELOPMENT.md) | 开发环境、最短跑通路径、常见路径索引 |
| [API-OVERVIEW.md](./API-OVERVIEW.md) | HTTP 端点索引、iOS 最小 API 子集、`/api/v1` vs 无版本路径 |
| [API-CHANGELOG.md](./API-CHANGELOG.md) | 对外 JSON 契约变更记录（维护约定） |
| [API.md](./API.md) | REST / SSE 详解与 Swift 示例（与运行中 `/docs`、`openapi.json` 对照） |
| [MULTI_PLATFORM.md](./MULTI_PLATFORM.md) | 多端对齐：后端规范 8 路 RMS、会话归档、手机↔网页互通 |
| [MODEL_TRAINING_GUIDE.md](./MODEL_TRAINING_GUIDE.md) | 模型训练相关 |

## 上架与产品（FocuX）

- [`../ios/Distribution/appstore-metadata.md`](../ios/Distribution/appstore-metadata.md) — App Store Connect 元数据备忘

## 可选 / 专题（非主线交付）

- [LeLamp-FluxChi-HCI-HRI.md](./LeLamp-FluxChi-HCI-HRI.md) — 与第三方表达机器人、ELEGNT 等扩展思路（**当前产品迭代可不读**）
- [RESEARCH-Camera-Vision.md](./RESEARCH-Camera-Vision.md) — 摄像头 + EMG 多模态调研笔记（内部路线图）
- [VISION-WEBSOCKET-SPEC.md](./VISION-WEBSOCKET-SPEC.md) — `/ws/vision` 客户端 JSON 格式（实现契约）
