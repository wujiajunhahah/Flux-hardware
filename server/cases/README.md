# Contract Cases

This directory holds the FluxChi platform contract harness inputs.

## Layout

```text
server/cases/
  contracts/        # checked-in regression cases
  fixtures/         # larger payload fixtures referenced by cases
  captured/         # sanitized captured failures, disabled by default
  targets.example.json
```

## Run the Core Contract Suite

Install dev dependencies:

```bash
python3 -m pip install -r server/requirements-dev.txt
```

Run the staging core chain:

```bash
python3 -m server.scripts.run_contract_cases --target staging --tag core
```

Or use the unified platform check entrypoint:

```bash
python3 -m server.scripts.run_platform_checks
```

Run one specific case:

```bash
python3 -m server.scripts.run_contract_cases --target staging --case session_upload_requires_https_url_v1
```

## Targets File

`targets.example.json` is only a template. Real live runs should use a server-local or CI-provided targets file.

Recommended local flow:

```bash
cp server/cases/targets.example.json server/cases/targets.local.json
```

Then either edit `server/cases/targets.local.json` manually, or let the helper script prepare fresh shadow auth material for you:

```bash
python3 -m server.scripts.prepare_shadow_target \
  --target production \
  --targets-path server/cases/targets.local.json \
  --provider-token dev:shadow-production
```

After that, run:

```bash
python3 -m server.scripts.run_contract_cases --target production --tag shadow --targets-path server/cases/targets.local.json
```

Or run the same shadow path through the unified platform check entrypoint:

```bash
python3 -m server.scripts.run_platform_checks \
  --skip-staging-core \
  --production-shadow \
  --targets-path server/cases/targets.local.json
```

For GitHub Actions, the same target profile JSON should be stored in the `FLUX_CONTRACT_TARGETS_JSON` secret. If you also configure `FLUX_CONTRACT_PRODUCTION_PROVIDER_TOKEN`, the scheduled production shadow job will refresh the production `Authorization` header and `device_id` automatically before running.

`server/cases/targets.local.json` is gitignored and is the easiest place to hand over a live targets file during local/server work.

Supported target profile fields:

- `base_url`
- `allow_writes`
- `allow_capture`
- `allow_shadow_read`
- `default_headers`
- `seed_variables`

`default_headers` is useful for read-only shadow runs where the API token is injected outside the case file.

`seed_variables` is useful for values like `device_id` that should exist before the first step starts.

Minimal production shadow example:

```json
{
  "production": {
    "base_url": "https://api.focux.me",
    "allow_writes": false,
    "allow_capture": false,
    "allow_shadow_read": true,
    "default_headers": {
      "Authorization": "Bearer <read-only-token>"
    },
    "seed_variables": {
      "device_id": "dev_replace_me"
    }
  }
}
```

## Artifacts

Run artifacts land under:

```text
artifacts/contracts/<run_id>/
```

Important files:

- `events.jsonl`
- `cases/<case_id>/steps/<step_id>/request.json`
- `cases/<case_id>/steps/<step_id>/response.json`
- `cases/<case_id>/steps/<step_id>/session_blob.json`
- `cases/<case_id>/steps/<step_id>/finalize_payload.json`
- `cases/<case_id>/steps/<step_id>/feedback_payload.json`
- `cases/<case_id>/steps/<step_id>/step_trace.json`

## Captured Failures

Convert a sanitized failure bundle into a disabled captured case:

```bash
python3 -m server.scripts.capture_contract_failure \
  --input /path/to/failure-bundle.json \
  --output-dir server/cases/captured
```

Captured cases are:

- `source: captured`
- `enabled: false`
- redacted before they are written
- not promoted into `contracts/` automatically

Promotion into `server/cases/contracts/` stays an explicit manual action.

## Production Shadow

Production replay is read-only by default. Mutating cases will fail without `--break-glass`.

Use the shadow tag for read-only checks:

```bash
python3 -m server.scripts.run_contract_cases --target production --tag shadow --targets-path server/cases/targets.local.json
```

For a production shadow target to work, its target profile must provide:

- a valid read-only bearer token via `default_headers.Authorization`
- `device_id` in `seed_variables`
