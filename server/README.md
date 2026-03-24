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

Recommended server bootstrap flow from the repo root:

```bash
python3 -m venv .venv && .venv/bin/pip install -r server/requirements.txt
```

Example `~/fluxchi-app.env`:

```text
FLUX_ENV=production
FLUX_SECRET_KEY=replace-with-a-random-secret
FLUX_BIND_HOST=127.0.0.1
FLUX_PORT=8000
FLUX_ALLOW_INSECURE_PROVIDER_TOKENS=false
```

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
