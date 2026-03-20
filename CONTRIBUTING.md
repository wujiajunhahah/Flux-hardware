# Contributing to FluxChi

## Scope

- Prefer **small, focused PRs** tied to an issue or a clear bugfix/feature.
- Match existing style in the touched area (Swift / Python / docs).

## Secrets

**Never commit:**

- Discord webhooks, API keys, provisioning profiles, signing secrets
- Personal IPs or machine-specific paths in **shared** docs (local notes are OK in your own branch)

Use environment variables or CI **Secrets**.

## API changes

- New or changed **`/api/v1/*` JSON** shapes should be noted in [`docs/API-CHANGELOG.md`](docs/API-CHANGELOG.md).
- Update [`docs/API.md`](docs/API.md) when behavior is user-visible; cross-check **Swagger** (`/docs`) after changing `web/app.py`.

## iOS

- Bump **`MARKETING_VERSION` / `CURRENT_PROJECT_VERSION`** only when preparing a release; keep [`ios/FluxChi/Models/FluxMeta.swift`](ios/FluxChi/Models/FluxMeta.swift) `version` in sync for in-app display.

## License

Academic project -- Shenzhen Technology University (see root [README](README.md)).
