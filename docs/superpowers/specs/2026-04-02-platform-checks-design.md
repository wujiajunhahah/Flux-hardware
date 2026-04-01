# Platform Checks Design

## Goal

Add one stable backend-facing validation entrypoint so any client integration path can run the same platform checks before relying on the shared API surface.

## Scope

This design adds a thin orchestration layer on top of the existing contract harness.

Included:

- run `staging` core contract checks from one command
- optionally refresh production shadow auth material before running shadow checks
- optionally run production shadow checks from the same command
- emit one aggregated JSON summary for local, server-side, and CI usage

Excluded:

- new backend API routes
- new contract cases
- WebSocket control planes
- simulator or UI automation

## Approach

Create a small orchestration module under `server/app/testing/` that coordinates existing helpers:

- `run_cases(...)` for contract execution
- `prepare_shadow_target(...)` and `update_target_profile_auth(...)` for production shadow auth refresh

Expose this through a new script entrypoint:

- `python3 -m server.scripts.run_platform_checks`

The script should support three practical modes:

1. `staging-core` only
2. `staging-core + production-shadow`
3. `production-shadow` only

Production shadow preparation stays explicit via CLI flags so the user controls when live auth material is refreshed.

## Output Contract

The script returns one JSON object with:

- top-level `ok`
- selected check names
- per-check summaries
- prepared shadow auth metadata when used

This gives downstream automation one stable result shape instead of parsing multiple command outputs.

## Error Handling

- Missing required shadow prep input should fail fast with a clear CLI error
- staging failure should mark the whole run failed
- production shadow failure should also mark the run failed when selected
- no best-effort silent skipping

## Testing

Add focused unit tests that verify:

- staging-only mode calls the contract harness once
- shadow mode optionally refreshes auth then runs production shadow
- aggregated summary shape is stable
- CLI argument validation rejects invalid flag combinations
