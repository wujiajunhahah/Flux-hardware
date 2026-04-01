# Server Skeleton

This directory is the Phase-1 backend foundation for the modular monolith described in:

- `docs/ARCHITECTURE-PRODUCTION-V1.md`
- `docs/PLATFORM-CONTRACT-V1.md`

It does not replace `web/app.py` yet.

## Current scope

- Base FastAPI app entry
- Request ID middleware and uniform API envelope
- Auth routes: `sign-up`, `sign-in`, `refresh`, `sign-out`
- Devices routes: list and revoke
- Profile routes: get/update profile and per-device calibration
- Environment-driven settings bootstrap
- Lightweight SQL migration runner with schema tracking
- Initial Postgres schema migration

## Setup

Install the backend dependencies before enabling database-backed routes:

```bash
python3 -m pip install -r server/requirements.txt
```

Copy `server/.env.example` to `server/.env` and adjust values as needed.

Primary settings:

- `FLUX_ENV`
- `FLUX_DATABASE_URL`
- `FLUX_SECRET_KEY`
- `FLUX_ACCESS_TOKEN_TTL_SEC`
- `FLUX_REFRESH_TOKEN_TTL_SEC`
- `FLUX_UPLOAD_URL_TTL_SEC`
- `FLUX_DEVICE_LIMIT`
- `FLUX_ACCESS_TOKEN_ALGORITHM`
- `FLUX_ACCESS_TOKEN_ISSUER`
- `FLUX_ACCESS_TOKEN_AUDIENCE`
- `FLUX_ALLOW_INSECURE_PROVIDER_TOKENS`

The settings loader checks both repo-root `.env` and `server/.env`.

## Run the skeleton app

```bash
python3 -m uvicorn server.app.main:app --reload
```

Default health check:

```text
GET /v1/health
```

Implemented routes:

```text
POST /v1/auth/sign-up
POST /v1/auth/sign-in
POST /v1/auth/refresh
POST /v1/auth/sign-out
GET  /v1/devices
DELETE /v1/devices/{device_id}
GET  /v1/profile
PUT  /v1/profile
GET  /v1/devices/{device_id}/calibration
PUT  /v1/devices/{device_id}/calibration
```

## Migration strategy

SQL migrations live under `server/migrations/`.

Applied migrations are tracked in Postgres via `schema_migrations`:

- `filename`
- `checksum_sha256`
- `applied_at`

Run migrations:

```bash
python3 -m server.scripts.migrate
```

Inspect migration status:

```bash
python3 -m server.scripts.migrate --status
```

The runner fails fast if an already-applied migration file is edited and its checksum no longer matches.

## ECS deployment with server-local Postgres

The deployment scripts under `server/scripts/` are designed for the Alibaba ECS setup where Postgres listens on `127.0.0.1:5432` and secrets stay in server-local files that are not committed.

Expected server-local env files:

- `~/fluxchi-db.env`: database secret such as `FLUX_DATABASE_URL=...`
- `~/fluxchi-app.env`: app runtime settings such as `FLUX_ENV`, `FLUX_SECRET_KEY`, `FLUX_PORT`

The helper scripts auto-load both files before starting:

```bash
bash server/scripts/run_migrations.sh
bash server/scripts/run_api.sh
```

Setup helpers:

```bash
bash server/scripts/install_systemd_service.sh
bash server/scripts/install_nginx_proxy.sh
.venv/bin/python -m server.scripts.smoke_test
```

## Contract Harness

The server now also includes a staging-first contract regression harness for the platform chain.

Install dev dependencies:

```bash
python3 -m pip install -r server/requirements-dev.txt
```

Run the core regression gate:

```bash
python3 -m server.scripts.run_contract_cases --target staging --tag core
```

Run the unified platform check entrypoint for the default staging-core gate:

```bash
python3 -m server.scripts.run_platform_checks
```

Run a single regression case:

```bash
python3 -m server.scripts.run_contract_cases --target staging --case session_upload_requires_https_url_v1
```

Read-only production shadow checks use the `shadow` tag and a live targets file with read-only auth material:

```bash
cp server/cases/targets.example.json server/cases/targets.local.json
python3 -m server.scripts.prepare_shadow_target \
  --target production \
  --targets-path server/cases/targets.local.json \
  --provider-token dev:shadow-production
python3 -m server.scripts.run_contract_cases --target production --tag shadow --targets-path server/cases/targets.local.json
```

Or run the same production shadow check through the unified entrypoint:

```bash
python3 -m server.scripts.run_platform_checks \
  --skip-staging-core \
  --production-shadow \
  --targets-path server/cases/targets.local.json
```

## GitHub Actions Setup

The contract workflow lives at `.github/workflows/contract-cases.yml`.

Required repository variables:

- `FLUX_CONTRACT_STAGING_ENABLED=true` to enable the staging core gate on push and pull request
- `FLUX_CONTRACT_PRODUCTION_SHADOW_ENABLED=true` to enable scheduled or manual production shadow checks

Required repository secret:

- `FLUX_CONTRACT_TARGETS_JSON`
  - must be one JSON object containing every target profile the workflow needs
  - staging CI needs a writable `staging` profile
  - production shadow needs a read-only `production` profile

Recommended production shadow secret:

- `FLUX_CONTRACT_PRODUCTION_PROVIDER_TOKEN`
  - if present, the workflow refreshes production shadow `Authorization` and `device_id` automatically by running `python3 -m server.scripts.prepare_shadow_target`
  - this avoids storing a short-lived `access_token` directly in GitHub secrets

Minimal CI secret shape:

```json
{
  "staging": {
    "base_url": "https://staging.api.focux.me",
    "allow_writes": true,
    "allow_capture": true,
    "allow_shadow_read": false
  },
  "production": {
    "base_url": "https://api.focux.me",
    "allow_writes": false,
    "allow_capture": false,
    "allow_shadow_read": true,
    "default_headers": {
      "Authorization": "Bearer <fallback-read-only-token>"
    },
    "seed_variables": {
      "device_id": "dev_replace_me"
    }
  }
}
```

If `FLUX_CONTRACT_PRODUCTION_PROVIDER_TOKEN` is configured, the workflow overwrites the production `Authorization` and `seed_variables.device_id` fields at runtime before running the shadow check.

`server/cases/targets.local.json` is gitignored and is the recommended handoff path for live target config during local or server-side runs.

The checked-in case registry and capture workflow are documented in `server/cases/README.md`.

Artifacts land under `artifacts/contracts/<run_id>/`.

If you have a sanitized production failure bundle, convert it into a disabled captured case with:

```bash
python3 -m server.scripts.capture_contract_failure \
  --input /path/to/failure-bundle.json \
  --output-dir server/cases/captured
```

The old smoke script still exists for bring-up:

```bash
python3 -m server.scripts.smoke_test
```

Use the smoke script only for quick bring-up of auth, device/profile, calibration, model manifest, and bootstrap. Use contract cases for session upload, blob download, feedback, and release-gating regression coverage.

Recommended server bootstrap flow from the repo root:

```bash
python3 -m venv .venv && .venv/bin/pip install -r server/requirements.txt
```

Example `~/fluxchi-app.env`:

```text
FLUX_ENV=production
FLUX_SECRET_KEY=replace-with-a-random-secret
FLUX_PUBLIC_BASE_URL=https://api.focux.me
FLUX_BIND_HOST=127.0.0.1
FLUX_PORT=8000
FLUX_ALLOW_INSECURE_PROVIDER_TOKENS=false
```

Set `FLUX_PUBLIC_BASE_URL` whenever the API sits behind an HTTPS reverse proxy and the upstream app still listens on local plain HTTP. This makes generated absolute URLs such as `upload_url`, `download_url`, and model artifact URLs stable even if the proxy headers are incomplete.

If the current client is still using development passthrough provider tokens instead of real provider JWT verification, set `FLUX_ALLOW_INSECURE_PROVIDER_TOKENS=true` temporarily during bring-up.

Manual run:

```bash
bash server/scripts/run_migrations.sh
bash server/scripts/run_api.sh
```

Systemd example:

- copy `server/deploy/fluxchi-api.service.example` to `/etc/systemd/system/fluxchi-api.service`
- update the repo path if it is not `/home/admin/fluxchi`
- start with `sudo systemctl enable --now fluxchi-api`
- inspect with `sudo systemctl status fluxchi-api --no-pager -l`

## Notes

- The new server stack is intentionally isolated from the current local gateway.
- The gateway in `web/app.py` remains the current prototype/runtime path.
- Product API implementation should move into `server/app/` domain by domain.
- `psycopg` and `PyJWT` are imported lazily in the new infrastructure modules so the app skeleton can still boot before auth and database dependencies are installed.
- Provider token resolution currently supports two development paths:
  - decode JWT `sub` without signature verification
  - fallback passthrough tokens when `FLUX_ALLOW_INSECURE_PROVIDER_TOKENS=true`
- Production should replace the development fallback with real provider verification.
