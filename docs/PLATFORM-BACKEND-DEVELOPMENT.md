# FluxChi 平台后端开发手册

这份文档是 **多端共用后端** 的开发与运维入口。目标不是解释全部历史，而是给后续维护时一个稳定的起点：

- 本地怎么跑
- staging / production 怎么验
- CI 怎么开
- 出问题先看哪里
- 哪些文件是平台入口，哪些只是辅助脚本

如果后续接入的不只是 iOS / Web / 硬件，而是更多客户端，这份文档也继续适用，因为它围绕的是 **统一平台后端边界**，不是某个单端实现。

---

## 1. 当前定位

现在的后端已经不是“几个 API 能跑”，而是一个开始具备平台形态的底座：

- 有正式 HTTPS 入口：`https://api.focux.me`
- 有本地 bring-up smoke：`server/scripts/smoke_test.py`
- 有可回归的 contract harness：`server/scripts/run_contract_cases.py`
- 有 production shadow 检查
- 有 shadow auth material 自动准备脚本
- 有统一的平台检查入口：`server/scripts/run_platform_checks.py`
- 有对应 GitHub Actions workflow

一句话：

> 后续无论接什么端，都应该先围绕这套平台检查入口验证，而不是每个端各自发明一套后端联调流程。

---

## 2. 关键入口文件

### 平台 API 与运行

- `server/app/main.py`：FastAPI 应用入口
- `server/app/api/`：路由层
- `server/app/services/`：业务逻辑
- `server/app/repositories/`：数据访问

### 平台验证与回归

- `server/scripts/smoke_test.py`
  - 只做 bring-up
  - 用于快速确认 `auth -> device/profile -> calibration -> manifest/bootstrap -> refresh/sign-out`
- `server/scripts/run_contract_cases.py`
  - 直接跑 contract harness
  - 用于指定 `target/tag/case` 的平台回归
- `server/scripts/prepare_shadow_target.py`
  - 为 production shadow 自动准备 `Authorization` 和 `device_id`
- `server/scripts/run_platform_checks.py`
  - 统一平台检查入口
  - 默认跑 staging core
  - 可选跑 production shadow

### contract harness 内核

- `server/app/testing/runner.py`
- `server/app/testing/transport.py`
- `server/app/testing/target_prep.py`
- `server/app/testing/platform_checks.py`
- `server/cases/contracts/`
- `server/cases/fixtures/`

---

## 3. 本地最小工作流

先安装依赖：

```bash
cd /path/to/harward-gesture
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r server/requirements-dev.txt
```

### 3.1 快速 bring-up

```bash
python3 -m server.scripts.smoke_test
```

适用场景：

- 新机器刚起服务
- 环境变量刚调整
- 先确认最小平台主路径没死

### 3.2 跑 staging core

```bash
python3 -m server.scripts.run_platform_checks
```

这会走统一入口，默认执行 staging core 平台检查。

如果只想直接跑 harness：

```bash
python3 -m server.scripts.run_contract_cases --target staging --tag core
```

### 3.3 跑 production shadow

先准备本地 live targets：

```bash
cp server/cases/targets.example.json server/cases/targets.local.json
```

然后自动刷新 production shadow auth material：

```bash
python3 -m server.scripts.prepare_shadow_target \
  --target production \
  --targets-path server/cases/targets.local.json \
  --provider-token dev:shadow-production
```

再跑 production shadow：

```bash
python3 -m server.scripts.run_platform_checks \
  --skip-staging-core \
  --production-shadow \
  --targets-path server/cases/targets.local.json
```

---

## 4. targets 文件规则

模板文件：

- `server/cases/targets.example.json`

本地敏感文件：

- `server/cases/targets.local.json`

规则：

- `targets.example.json` 只放结构，不放真实敏感值
- `targets.local.json` 是 gitignored
- 真实 live 运行都显式传 `--targets-path`

常用字段：

- `base_url`
- `allow_writes`
- `allow_capture`
- `allow_shadow_read`
- `default_headers`
- `seed_variables`

production shadow 至少需要：

- `default_headers.Authorization`
- `seed_variables.device_id`

如果用了 `prepare_shadow_target.py`，这两个字段会在运行前被自动更新。

---

## 5. GitHub Actions 启用方式

workflow：

- `.github/workflows/contract-cases.yml`

### Repository variables

- `FLUX_CONTRACT_STAGING_ENABLED=true`
- `FLUX_CONTRACT_PRODUCTION_SHADOW_ENABLED=true`

### Repository secrets

必须：

- `FLUX_CONTRACT_TARGETS_JSON`

推荐：

- `FLUX_CONTRACT_PRODUCTION_PROVIDER_TOKEN`

说明：

- `FLUX_CONTRACT_TARGETS_JSON` 提供 staging / production 两个 target profile
- 如果配置了 `FLUX_CONTRACT_PRODUCTION_PROVIDER_TOKEN`，CI 会先自动刷新 production shadow 的 `Authorization` 与 `device_id`
- 这能避免长期把会过期的 `access_token` 塞进 GitHub secret

### CI 当前行为

- `contract-unit`
  - 跑 `server/tests/testing`
- `contract-staging-core`
  - push / PR 时可启用
  - 调统一入口 `python -m server.scripts.run_platform_checks`
- `contract-production-shadow`
  - `schedule` / `workflow_dispatch` 时可启用
  - 先可选刷新 shadow auth，再跑统一入口

---

## 6. 常见排障顺序

### 情况 A：本地跑不通

先看：

1. `python3 -m server.scripts.smoke_test`
2. `python3 -m server.scripts.run_platform_checks`
3. `server/tests/testing`

如果是 HTTPS 相关：

- 先确认 `transport.py` 的 CA bundle 路径是否生效
- 优先用 `certifi` 或 `SSL_CERT_FILE`
- 不要直接关 TLS 校验

### 情况 B：production shadow 失败

先区分：

1. 本地网络 / DNS / CA 问题
2. token 过期
3. 后端真实回归

优先看：

- `artifacts/contracts/<run_id>/run.json`
- `artifacts/contracts/<run_id>/events.jsonl`
- `cases/<case_id>/summary.json`

### 情况 C：CI 失败

先查：

1. Repository variables 是否开启
2. `FLUX_CONTRACT_TARGETS_JSON` 是否是合法 JSON
3. production shadow 是否缺 `FLUX_CONTRACT_PRODUCTION_PROVIDER_TOKEN`
4. artifact 上传结果里具体失败在哪个 case / step

---

## 7. 当前边界

已经收口的：

- 平台 API 主链路
- HTTPS 入口
- smoke bring-up
- contract harness
- production shadow
- shadow target prep
- unified platform checks
- CI workflow 接统一入口

还没收口完的：

- staging 真正长期稳定运行
- CI 远端首次正式启用与观测
- iOS / Web / 硬件 全部接到同一平台回归节奏
- captured failure -> durable contract case 的更顺滑闭环

---

## 8. 后续维护原则

后续继续演进时，尽量保持这几个原则：

1. 新增后端能力，优先考虑是否要进入 platform checks
2. 新端接入前，先跑统一平台检查，而不是只看客户端表现
3. 出生产问题时，优先沉淀成 contract case 或 captured failure
4. 不要把 smoke script 又扩回完整链路
5. production shadow 默认保持只读

---

## 9. 一条推荐命令链

本地开发时，最常用的一条顺序通常是：

```bash
python3 -m server.scripts.smoke_test
python3 -m server.scripts.run_platform_checks
python3 -m pytest server/tests/testing -q
```

如果要验 production shadow：

```bash
python3 -m server.scripts.prepare_shadow_target \
  --target production \
  --targets-path server/cases/targets.local.json \
  --provider-token dev:shadow-production

python3 -m server.scripts.run_platform_checks \
  --skip-staging-core \
  --production-shadow \
  --targets-path server/cases/targets.local.json
```

这就是当前这套平台后端最稳定的维护入口。
