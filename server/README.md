# Server Skeleton

This directory is the Phase-1 backend foundation for the modular monolith described in:

- `docs/ARCHITECTURE-PRODUCTION-V1.md`
- `docs/PLATFORM-CONTRACT-V1.md`

It does not replace `web/app.py` yet.

## Current scope

- Base FastAPI app entry
- Router skeleton
- Environment-driven settings bootstrap
- Lightweight SQL migration runner with schema tracking
- Initial Postgres schema migration

## Setup

Install the backend dependencies before enabling database-backed routes:

```bash
python3 -m pip install -r requirements.txt
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

The settings loader checks both repo-root `.env` and `server/.env`.

## Run the skeleton app

```bash
uvicorn server.app.main:app --reload
```

Default health check:

```text
GET /v1/health
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

## Notes

- The new server stack is intentionally isolated from the current local gateway.
- The gateway in `web/app.py` remains the current prototype/runtime path.
- Product API implementation should move into `server/app/` domain by domain.
- `psycopg` and `PyJWT` are imported lazily in the new infrastructure modules so the app skeleton can still boot before auth and database dependencies are installed.
