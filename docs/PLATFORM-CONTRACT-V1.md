# FluxChi 平台合同 V1

> 目的：定义 FluxChi 面向真实产品形态的 **API Contract + Data Model + 约束规则**。  
> 范围：在线控制面，不覆盖端上实时推理实现细节。  
> 状态：V1 草案，供后端/iOS/算法/产品共同审校。  
> 关联文档：目标架构见 [ARCHITECTURE-PRODUCTION-V1.md](./ARCHITECTURE-PRODUCTION-V1.md)。

---

## 1. 本文钉死的决策

V1 直接定死，不再写成“可选方案”：

1. `feedback_event` 是独立追加写资源，**不**塞进 `profile_state`
2. `profile_state` 只存当前态和摘要统计，写入策略为 **`base_version + 409 Conflict`**
3. `session blob` 采用 **三段式上传**
   - `POST /v1/sessions`
   - `PUT upload_url`
   - `PATCH /v1/sessions/{id}`
4. `GET /v1/sync/bootstrap` 是聚合读，V1 明确只返回：
   - `profile_state`
   - `device_calibrations`
   - `active_model_manifest`
   - `server_time`
5. 单用户最多 **5 台活跃设备**
   - 超限返回 `409 device_limit_exceeded`
   - 必须显式 `DELETE /v1/devices/{device_id}` 释放名额

---

## 2. 作用域与非目标

### 2.1 作用域

- 用户认证
- 设备绑定
- profile 当前态同步
- device calibration 同步
- feedback 事件写入
- session 元数据与 blob 上传
- active model manifest 下发
- bootstrap 聚合读取

### 2.2 非目标

- 训练任务内部调度细节
- 模型特征工程细节
- 端上实时采集协议
- 管理后台 UI
- 支付、订阅、计费

---

## 3. 全局约定

### 3.1 路径前缀

平台合同使用：

```text
/v1/*
```

说明：

- 当前本地 Python 网关仍使用 `/api/v1/*`
- 产品化在线控制面以本文的 `/v1/*` 为准
- 若未来边缘网关统一挂到 `/api/v1/*`，应做无歧义路径映射，不改字段契约

### 3.2 认证

除注册/登录外，其余请求默认需要：

```http
Authorization: Bearer <access_token>
```

### 3.3 时间

- 一律 ISO8601 UTC
- 示例：`2026-03-24T12:00:00Z`

### 3.4 ID 规则

| 资源 | 前缀 | 示例 |
|------|------|------|
| user | `usr_` | `usr_01H...` |
| profile | `pro_` | `pro_01H...` |
| device | `dev_` | `dev_01H...` |
| feedback_event | `fbk_` | `fbk_01H...` |
| session | `ses_` | `ses_01H...` |
| model_release | `mdl_` | `mdl_01H...` |

要求：

- 对外 API 只暴露稳定资源 ID
- 不暴露数据库自增 ID
- 不使用本地 hash 作为跨设备资源主键

### 3.5 响应信封

成功：

```json
{
  "ok": true,
  "request_id": "req_01H...",
  "data": {}
}
```

失败：

```json
{
  "ok": false,
  "request_id": "req_01H...",
  "error": {
    "code": "profile_version_conflict",
    "message": "base_version is stale"
  }
}
```

### 3.6 分页

列表接口默认游标分页：

```json
{
  "items": [],
  "next_cursor": "cur_..."
}
```

### 3.7 幂等

以下接口必须支持幂等：

- `POST /v1/feedback-events`
- `POST /v1/sessions`

Header:

```http
Idempotency-Key: <stable-client-key>
```

约定：

- 幂等 key 作用域为 `user_id + route`
- 重复请求返回第一次成功写入的结果

---

## 4. Auth 领域

### 4.1 接口

#### `POST /v1/auth/sign-up`

请求：

```json
{
  "provider": "apple",
  "provider_token": "...",
  "device": {
    "client_device_key": "ios-installation-uuid",
    "platform": "ios",
    "device_name": "Jiajun iPhone",
    "app_version": "1.3.0",
    "os_version": "iOS 26.0"
  }
}
```

响应：

```json
{
  "ok": true,
  "request_id": "req_...",
  "data": {
    "user_id": "usr_...",
    "device_id": "dev_...",
    "access_token": "...",
    "refresh_token": "...",
    "expires_in_sec": 3600
  }
}
```

#### `POST /v1/auth/sign-in`

与 `sign-up` 相同。  
语义：

- `sign-up` 只用于创建新用户，不做 upsert
- `sign-in` 只用于已存在用户登录
- 若 identity 不存在，`sign-in` 返回 `404 user_not_found`
- 客户端在 onboarding 完成前调用 `sign-up`，之后统一调用 `sign-in`

若这是新设备登录且用户活跃设备数已达上限，返回：

```json
{
  "ok": false,
  "request_id": "req_...",
  "error": {
    "code": "device_limit_exceeded",
    "message": "maximum active devices reached"
  }
}
```

#### `POST /v1/auth/refresh`

请求：

```json
{
  "refresh_token": "..."
}
```

响应：

```json
{
  "ok": true,
  "request_id": "req_...",
  "data": {
    "access_token": "...",
    "expires_in_sec": 3600
  }
}
```

#### `POST /v1/auth/sign-out`

请求：

```json
{
  "refresh_token": "..."
}
```

行为：

- 撤销当前 refresh token
- 不自动删除 `device_binding`

---

### 4.2 数据表

#### `users`

| 字段 | 类型 | 约束 |
|------|------|------|
| `user_id` | text | PK |
| `status` | text | `active` / `disabled` / `deleted` |
| `primary_email` | text | nullable |
| `timezone` | text | nullable |
| `created_at` | timestamptz | not null |
| `updated_at` | timestamptz | not null |

索引：

- PK(`user_id`)

#### `auth_identities`

| 字段 | 类型 | 约束 |
|------|------|------|
| `identity_id` | text | PK |
| `user_id` | text | FK -> `users.user_id` |
| `provider` | text | not null |
| `provider_subject` | text | not null |
| `created_at` | timestamptz | not null |

约束：

- UNIQUE(`provider`, `provider_subject`)

#### `refresh_tokens`

| 字段 | 类型 | 约束 |
|------|------|------|
| `token_id` | text | PK |
| `user_id` | text | FK |
| `device_id` | text | FK -> `device_bindings.device_id` |
| `token_hash` | text | unique |
| `expires_at` | timestamptz | not null |
| `revoked_at` | timestamptz | nullable |
| `created_at` | timestamptz | not null |

索引：

- UNIQUE(`token_hash`)
- INDEX(`user_id`, `device_id`)

#### `device_bindings`

| 字段 | 类型 | 约束 |
|------|------|------|
| `device_id` | text | PK |
| `user_id` | text | FK |
| `client_device_key` | text | not null |
| `platform` | text | `ios` / `android` / `web` |
| `device_name` | text | not null |
| `app_version` | text | nullable |
| `os_version` | text | nullable |
| `last_seen_at` | timestamptz | not null |
| `revoked_at` | timestamptz | nullable |
| `created_at` | timestamptz | not null |

服务约束：

- 活跃设备定义：`revoked_at IS NULL`
- 单用户最多 5 条活跃设备记录
- 同一用户下，`client_device_key` 必须稳定标识同一次 App 安装实例
- 同一用户重复登录同一安装实例时，复用既有 `device_binding`，不新建设备记录

索引：

- INDEX(`user_id`, `revoked_at`)
- UNIQUE(`user_id`, `client_device_key`)

---

### 4.3 Auth / Devices 补充约束

- `sign-up` 命中已存在的 `(provider, provider_subject)` 时返回 `409 identity_already_exists`
- `sign-in` 命中不存在的 `(provider, provider_subject)` 时返回 `404 user_not_found`
- 设备上限按活跃设备数计算，不区分平台
- `DELETE /v1/devices/{device_id}` 是释放设备名额的唯一正式入口

---

## 5. Devices 领域

### 5.1 接口

#### `GET /v1/devices`

响应：

```json
{
  "ok": true,
  "request_id": "req_...",
  "data": {
    "items": [
      {
        "device_id": "dev_...",
        "platform": "ios",
        "device_name": "Jiajun iPhone",
        "app_version": "1.3.0",
        "last_seen_at": "2026-03-24T12:00:00Z",
        "is_current_device": true
      }
    ]
  }
}
```

#### `DELETE /v1/devices/{device_id}`

行为：

- 标记 `device_binding.revoked_at`
- 使该设备 refresh token 全部失效
- 不删除历史 session / feedback

错误：

- `404 device_not_found`
- `409 cannot_delete_last_active_device` 可选，若产品决定至少保留一台

---

### 5.2 与 profile 的关系

- `device_binding` 代表账号级设备注册关系
- `device_calibration` 代表该设备上的个性化校准状态
- 一个设备可以被撤销绑定，但历史校准与会话数据仍可保留作审计/回溯

---

## 6. Profile 领域

### 6.1 接口

#### `GET /v1/profile`

响应：

```json
{
  "ok": true,
  "request_id": "req_...",
  "data": {
    "profile_state": {
      "profile_id": "pro_...",
      "user_id": "usr_...",
      "version": 12,
      "calibration_offset": 6.5,
      "estimated_accuracy": 82.0,
      "training_count": 143,
      "active_model_release_id": "mdl_...",
      "summary": {
        "retained_feedback_count": 500,
        "avg_absolute_error": 11.2,
        "last_feedback_at": "2026-03-24T11:59:00Z"
      },
      "updated_at": "2026-03-24T12:00:00Z"
    }
  }
}
```

#### `PUT /v1/profile`

请求：

```json
{
  "base_version": 12,
  "calibration_offset": 7.1,
  "estimated_accuracy": 83.5,
  "training_count": 144,
  "active_model_release_id": "mdl_...",
  "summary": {
    "retained_feedback_count": 500,
    "avg_absolute_error": 10.9,
    "last_feedback_at": "2026-03-24T12:05:00Z"
  }
}
```

成功响应：

```json
{
  "ok": true,
  "request_id": "req_...",
  "data": {
    "profile_state": {
      "profile_id": "pro_...",
      "user_id": "usr_...",
      "version": 13,
      "calibration_offset": 7.1,
      "estimated_accuracy": 83.5,
      "training_count": 144,
      "active_model_release_id": "mdl_...",
      "summary": {
        "retained_feedback_count": 500,
        "avg_absolute_error": 10.9,
        "last_feedback_at": "2026-03-24T12:05:00Z"
      },
      "updated_at": "2026-03-24T12:05:01Z"
    }
  }
}
```

冲突响应：

```json
{
  "ok": false,
  "request_id": "req_...",
  "error": {
    "code": "profile_version_conflict",
    "message": "base_version is stale"
  },
  "data": {
    "current_profile_state": {
      "profile_id": "pro_...",
      "version": 13,
      "updated_at": "2026-03-24T12:06:00Z"
    }
  }
}
```

规则：

- V1 不用 `updated_at` 冲突判定
- V1 直接使用 `base_version`
- 服务端成功写入后版本 `+1`

#### `GET /v1/devices/{device_id}/calibration`

响应：

```json
{
  "ok": true,
  "request_id": "req_...",
  "data": {
    "device_calibration": {
      "device_id": "dev_...",
      "user_id": "usr_...",
      "version": 3,
      "device_name": "Jiajun iPhone",
      "sensor_profile": {
        "source": "ble",
        "channel_count": 8
      },
      "calibration_offset": 5.0,
      "updated_at": "2026-03-24T12:00:00Z"
    }
  }
}
```

#### `PUT /v1/devices/{device_id}/calibration`

请求：

```json
{
  "base_version": 3,
  "device_name": "Jiajun iPhone",
  "sensor_profile": {
    "source": "ble",
    "channel_count": 8
  },
  "calibration_offset": 5.4
}
```

规则：

- 与 `PUT /v1/profile` 相同，使用 `base_version + 409`

---

### 6.2 数据表

#### `profile_states`

| 字段 | 类型 | 约束 |
|------|------|------|
| `profile_id` | text | PK |
| `user_id` | text | FK -> `users.user_id`, unique |
| `version` | integer | not null, `>= 1` |
| `calibration_offset` | numeric(6,2) | not null |
| `estimated_accuracy` | numeric(5,2) | not null, `0..100` |
| `training_count` | integer | not null, `>= 0` |
| `active_model_release_id` | text | nullable, FK -> `model_releases.model_release_id` |
| `summary_json` | jsonb | not null |
| `updated_at` | timestamptz | not null |
| `created_at` | timestamptz | not null |

索引：

- UNIQUE(`user_id`)
- INDEX(`active_model_release_id`)

说明：

- `summary_json` 只存摘要统计
- 不存 `feedbackPairs`

#### `device_calibrations`

| 字段 | 类型 | 约束 |
|------|------|------|
| `user_id` | text | FK |
| `device_id` | text | FK -> `device_bindings.device_id` |
| `version` | integer | not null, `>= 1` |
| `device_name` | text | not null |
| `sensor_profile_json` | jsonb | not null |
| `calibration_offset` | numeric(6,2) | not null |
| `updated_at` | timestamptz | not null |
| `created_at` | timestamptz | not null |

主键：

- PRIMARY KEY(`user_id`, `device_id`)

索引：

- INDEX(`device_id`)

说明：

- V1 **有意不引入**独立 `calibration_id`
- V1 把设备校准视为“每用户-每设备一条当前态”
- 若未来需要历史版本或审计，再追加 `device_calibration_revisions`

---

## 7. Feedback 领域

### 7.1 接口

#### `POST /v1/feedback-events`

Header：

```http
Idempotency-Key: ios:ses_01H...:feedback_v1
```

请求：

```json
{
  "feedback_event_id": "fbk_...",
  "device_id": "dev_...",
  "session_id": "ses_...",
  "predicted_stamina": 61,
  "actual_stamina": 35,
  "label": "fatigued",
  "kss": 8,
  "note": "felt exhausted after long session",
  "created_at": "2026-03-24T12:00:00Z"
}
```

成功响应：

```json
{
  "ok": true,
  "request_id": "req_...",
  "data": {
    "feedback_event": {
      "feedback_event_id": "fbk_...",
      "session_id": "ses_...",
      "label": "fatigued",
      "created_at": "2026-03-24T12:00:00Z"
    }
  }
}
```

规则：

- 这是追加写资源
- 若幂等 key 重复，返回第一次成功写入结果
- 写入成功后允许异步更新 `profile_state.summary`

#### `GET /v1/feedback-events`

查询参数：

- `cursor`
- `limit`
- `session_id` 可选

响应：

```json
{
  "ok": true,
  "request_id": "req_...",
  "data": {
    "items": [],
    "next_cursor": null
  }
}
```

---

### 7.2 数据表

#### `feedback_events`

| 字段 | 类型 | 约束 |
|------|------|------|
| `feedback_event_id` | text | PK |
| `user_id` | text | FK |
| `device_id` | text | FK |
| `session_id` | text | nullable, FK -> `sessions.session_id` |
| `idempotency_key` | text | not null |
| `predicted_stamina` | integer | `0..100` |
| `actual_stamina` | integer | `0..100` |
| `label` | text | `fatigued` / `alert` |
| `kss` | integer | nullable, `1..9` |
| `note` | text | nullable |
| `created_at` | timestamptz | not null |

约束：

- UNIQUE(`user_id`, `idempotency_key`)

索引：

- INDEX(`user_id`, `created_at DESC`)
- INDEX(`session_id`)
- INDEX(`label`, `created_at DESC`)

---

## 8. Session 领域

### 8.1 三段式上传

V1 统一采用三段式上传：

1. 客户端 `POST /v1/sessions` 创建上传会话
2. 客户端使用返回的 `upload_url` 直接 `PUT` blob 到对象存储
3. 客户端 `PATCH /v1/sessions/{session_id}` 将状态置为 `ready`

原因：

- App API 不承载大文件流量
- 更容易做 CDN / 对象存储治理
- 更利于后续服务拆分

### 8.2 接口

#### `POST /v1/sessions`

Header：

```http
Idempotency-Key: ios:ses_01H...:upload_v1
```

请求：

```json
{
  "session_id": "ses_...",
  "device_id": "dev_...",
  "source": "ios_ble",
  "title": "Morning Focus",
  "started_at": "2026-03-24T10:00:00Z",
  "ended_at": "2026-03-24T11:10:00Z",
  "duration_sec": 4200,
  "snapshot_count": 8400,
  "schema_version": 1,
  "content_type": "application/json",
  "size_bytes": 1843200,
  "sha256": "abc123..."
}
```

响应：

```json
{
  "ok": true,
  "request_id": "req_...",
  "data": {
    "session": {
      "session_id": "ses_...",
      "status": "pending_upload"
    },
    "upload": {
      "object_key": "sessions/2026/03/24/ses_....json",
      "upload_url": "https://object-store.example/...",
      "upload_method": "PUT",
      "content_type": "application/json",
      "expires_at": "2026-03-24T12:10:00Z"
    }
  }
}
```

#### `PATCH /v1/sessions/{session_id}`

请求：

```json
{
  "status": "ready",
  "blob": {
    "object_key": "sessions/2026/03/24/ses_....json",
    "size_bytes": 1843200,
    "sha256": "abc123..."
  }
}
```

成功响应：

```json
{
  "ok": true,
  "request_id": "req_...",
  "data": {
    "session": {
      "session_id": "ses_...",
      "status": "ready",
      "download_url": "/v1/sessions/ses_.../download"
    }
  }
}
```

错误：

- `409 session_not_uploaded`
- `409 session_checksum_mismatch`

#### `GET /v1/sessions`

查询参数：

- `cursor`
- `limit`
- `status` 可选

#### `GET /v1/sessions/{session_id}`

返回元数据，不返回 blob 正文。

#### `GET /v1/sessions/{session_id}/download`

返回短时效下载 URL：

```json
{
  "ok": true,
  "request_id": "req_...",
  "data": {
    "download_url": "https://object-store.example/...",
    "expires_at": "2026-03-24T12:15:00Z"
  }
}
```

---

### 8.3 数据表

#### `sessions`

| 字段 | 类型 | 约束 |
|------|------|------|
| `session_id` | text | PK |
| `user_id` | text | FK |
| `device_id` | text | FK |
| `idempotency_key` | text | not null |
| `status` | text | `pending_upload` / `ready` / `failed` / `deleted` |
| `source` | text | not null |
| `title` | text | nullable |
| `started_at` | timestamptz | not null |
| `ended_at` | timestamptz | not null |
| `duration_sec` | integer | not null |
| `snapshot_count` | integer | not null |
| `schema_version` | integer | not null |
| `content_type` | text | not null |
| `blob_object_key` | text | nullable |
| `blob_size_bytes` | bigint | nullable |
| `blob_sha256` | text | nullable |
| `created_at` | timestamptz | not null |
| `updated_at` | timestamptz | not null |

约束：

- UNIQUE(`user_id`, `idempotency_key`)

索引：

- INDEX(`user_id`, `started_at DESC`)
- INDEX(`device_id`, `started_at DESC`)
- INDEX(`status`, `updated_at DESC`)

---

## 9. Model 领域

### 9.1 接口

#### `GET /v1/models/manifest`

查询参数：

- `platform=ios`
- `channel=stable`
- `device_class=iphone`

响应：

```json
{
  "ok": true,
  "request_id": "req_...",
  "data": {
    "active_model_manifest": {
      "model_release_id": "mdl_...",
      "platform": "ios",
      "channel": "stable",
      "model_kind": "global_base_model",
      "version": "2026.03.24.1",
      "artifact_url": "https://object-store.example/models/....onnx",
      "artifact_sha256": "abc123...",
      "config": {
        "n_features": 84,
        "classes": ["alert", "fatigued"]
      },
      "published_at": "2026-03-24T12:00:00Z"
    }
  }
}
```

规则：

- V1 只发全局基础模型
- 用户个性化参数不随 manifest 下发

---

### 9.2 数据表

#### `model_releases`

| 字段 | 类型 | 约束 |
|------|------|------|
| `model_release_id` | text | PK |
| `model_kind` | text | `global_base_model` / `segment_model` |
| `platform` | text | not null |
| `channel` | text | `stable` / `beta` / `internal` |
| `version` | text | not null |
| `artifact_object_key` | text | not null |
| `artifact_sha256` | text | not null |
| `config_json` | jsonb | not null |
| `status` | text | `draft` / `published` / `retired` |
| `published_at` | timestamptz | nullable |
| `created_at` | timestamptz | not null |

约束：

- UNIQUE(`platform`, `channel`, `model_kind`, `version`)

索引：

- INDEX(`platform`, `channel`, `status`)

说明：

- 表内只存 `artifact_object_key`
- 接口响应中的 `artifact_url` 由服务端在读取时基于对象存储配置和短时效签名动态生成
- 禁止把完整外链 URL 直接落库到 `model_releases`

---

## 10. Sync 领域

### 10.1 V1 边界

V1 不做增量 `changes feed`。  
V1 只有一个聚合读取接口：

- `GET /v1/sync/bootstrap`

它的责任是让新设备或冷启动客户端一次性拿到“当前需要的最小控制面状态”。

### 10.2 接口

#### `GET /v1/sync/bootstrap`

查询参数：

- `device_id`
- `platform`
- `channel`

响应：

```json
{
  "ok": true,
  "request_id": "req_...",
  "data": {
    "server_time": "2026-03-24T12:00:00Z",
    "profile_state": {
      "profile_id": "pro_...",
      "version": 12,
      "calibration_offset": 6.5,
      "estimated_accuracy": 82.0,
      "training_count": 143,
      "active_model_release_id": "mdl_...",
      "summary": {
        "retained_feedback_count": 500,
        "avg_absolute_error": 11.2,
        "last_feedback_at": "2026-03-24T11:59:00Z"
      },
      "updated_at": "2026-03-24T12:00:00Z"
    },
    "device_calibrations": [
      {
        "device_id": "dev_...",
        "version": 3,
        "device_name": "Jiajun iPhone",
        "sensor_profile": {
          "source": "ble",
          "channel_count": 8
        },
        "calibration_offset": 5.0,
        "updated_at": "2026-03-24T11:58:00Z"
      }
    ],
    "active_model_manifest": {
      "model_release_id": "mdl_...",
      "platform": "ios",
      "channel": "stable",
      "model_kind": "global_base_model",
      "version": "2026.03.24.1",
      "artifact_url": "https://object-store.example/models/....onnx",
      "artifact_sha256": "abc123...",
      "config": {
        "n_features": 84,
        "classes": ["alert", "fatigued"]
      },
      "published_at": "2026-03-24T12:00:00Z"
    }
  }
}
```

规则：

- `bootstrap` 不返回 feedback 历史
- `bootstrap` 不返回 session 列表
- `bootstrap` 不负责增量游标推进

### 10.3 数据表

V1 无专属 `sync_*` 表。  
`bootstrap` 直接聚合自：

- `profile_states`
- `device_calibrations`
- `model_releases`

`sync_cursor` 进入 V2 再引入。

---

## 11. 跨领域约束

### 11.1 `profile_state` 与 `feedback_event`

- `feedback_event` 是训练标签源
- `profile_state.summary` 只是聚合结果
- 两者不是主从 blob 关系
- `feedback` 无上限地保留在事件流层
- `profile_state.summary.retained_feedback_count` 仅描述端上或服务端摘要窗口，不表示 feedback 被裁剪删除

### 11.2 `profile_state` 与 `device_calibration`

- `profile_state` 代表用户级当前态
- `device_calibration` 代表设备级修正
- 客户端展示值可以按产品策略组合：
  - `global base`
  - `profile offset`
  - `device calibration`

### 11.3 `session` 与 `feedback_event`

- `feedback_event.session_id` 允许为空，但推荐绑定
- 一场 session 可有多次反馈
- 训练构建时由 ETL 决定如何聚合

---

## 12. 错误码

| code | HTTP | 说明 |
|------|------|------|
| `unauthorized` | 401 | token 无效 |
| `forbidden` | 403 | 无访问权限 |
| `user_not_found` | 404 | `sign-in` 时身份不存在 |
| `identity_already_exists` | 409 | `sign-up` 时身份已存在 |
| `device_limit_exceeded` | 409 | 活跃设备数超限 |
| `profile_version_conflict` | 409 | `base_version` 过期 |
| `device_calibration_version_conflict` | 409 | 设备校准版本冲突 |
| `idempotency_key_conflict` | 409 | 同一路由复用了不兼容 payload 的幂等 key |
| `session_not_uploaded` | 409 | finalize 时 blob 不存在 |
| `session_checksum_mismatch` | 409 | 上传 blob 校验失败 |
| `resource_not_found` | 404 | 资源不存在 |
| `validation_error` | 422 | 请求字段非法 |
| `rate_limited` | 429 | 触发限流 |

### 12.1 V1 限流基线

V1 先采用固定量级限流，后续再按真实流量调优：

- `POST /v1/auth/sign-in`、`POST /v1/auth/sign-up`：每 IP 每分钟 10 次
- `POST /v1/feedback-events`：每用户每分钟 60 次
- `POST /v1/sessions`、`PATCH /v1/sessions/{session_id}`：每用户每分钟 30 次
- `GET /v1/profile`、`GET /v1/sync/bootstrap`、`GET /v1/models/manifest`：每用户每分钟 120 次

说明：

- 这些是 V1 的默认服务端限流配置，不是客户端重试建议
- 触发后统一返回 `429 rate_limited`

---

## 13. V1 实施顺序

按下面顺序做，返工最少：

1. Auth + Devices
2. Profile + Device Calibration
3. Feedback Events
4. Session 三段上传
5. Model Manifest
6. Sync Bootstrap

---

## 14. 与当前实现的映射

当前仓库已有的东西可以映射到未来领域，但不能直接等同：

| 当前实现 | 未来领域 |
|----------|----------|
| `src/profile_store.py` | `profile_state` 的过渡本地实现 |
| `PersonalizationManager.swift` | 端上 profile 应用层 |
| `src/flywheel/store.py` | `feedback_event + training data` 的本地过渡实现 |
| `data/sessions/*.json` | `session blob` 的本地过渡实现 |
| `web/app.py` | 未来模块化单体的原型入口 |

---

## 15. 下一步

本文定稿后，下一步不是继续写文档，而是：

1. 先搭数据库 schema migration
2. 再搭模块化单体骨架
3. 按领域逐个实现 V1 接口

在此之前，不建议直接把当前 `web/app.py` 大改成“伪微服务”结构。
