# FluxChi Contract Case Harness Design

> Date: 2026-04-01
> Status: Draft for review
> Owner: Codex
> Scope: Platform contract regression only. UI simulator automation is explicitly deferred.
> Milestone: V1A, core chain only

## Goal

Build a contract-case harness that turns the current platform happy path into a repeatable regression asset:

- manual happy-path cases for `bootstrap -> session upload -> feedback events`
- captured production failures that can be reviewed and promoted into replayable cases
- batch execution first
- event model that can later be streamed over WebSocket without rewriting the runner

## Context

The repository already has a real public platform path and a working happy-path smoke script:

- platform contract: [docs/PLATFORM-CONTRACT-V1.md](../../PLATFORM-CONTRACT-V1.md)
- current smoke runner: [server/scripts/smoke_test.py](../../../server/scripts/smoke_test.py)
- public URL generation fix: [server/app/core/public_url.py](../../../server/app/core/public_url.py)
- iOS client flows that exercise the same path:
  - [ios/FluxChi/Services/FluxService.swift](../../../ios/FluxChi/Services/FluxService.swift)
  - [ios/FluxChi/Services/ExportManager.swift](../../../ios/FluxChi/Services/ExportManager.swift)
  - [ios/FluxChi/Services/PersonalizationManager.swift](../../../ios/FluxChi/Services/PersonalizationManager.swift)
  - [ios/FluxChi/Views/SessionSummarySheet.swift](../../../ios/FluxChi/Views/SessionSummarySheet.swift)

The current repo does not yet have a dedicated `Tests` or `UITests` target for iOS automation. That makes simulator-driven end-to-end UI regression a later phase, not the starting point. The immediate value is to convert the public contract path into a case-driven runner with durable artifacts and failure replay.

## Decisions Already Locked

These decisions were made during the design conversation and are treated as V1A constraints:

1. Start with platform contract cases, not simulator-driven UI cases.
2. Use mixed case sources:
   - manually authored core happy-path cases
   - captured production failures that can be reviewed and promoted
3. Failure bundles store:
   - HTTP request and response
   - status code
   - key headers
   - bootstrap inputs
   - session blob payload
   - finalize payload
   - feedback payload
4. Default replay target is staging, not production.
5. Execution starts as batch CLI, but the protocol and event model must be WebSocket-ready.
6. V1A covers only the operations required by the initial six contract cases. Broader operation support is deferred to V1B.
7. V1A must become a real release guardrail:
   - core cases run in CI
   - failures notify a human owner
   - artifact retention is defined up front
8. Production target profiles are read-only by default. Mutating operations on production require an explicit break-glass override and are not part of the default regression path.

## Non-Goals

V1 explicitly does not include:

- direct simulator driving
- screenshot diffing
- full automatic promotion from production error to permanent regression set
- replacing existing business REST APIs with WebSocket
- generic test orchestration across unrelated product surfaces

## Product Shape

V1 has four layers:

1. `Case registry`
   - static contract case definitions
   - small values inline, large payloads in fixtures
2. `Runner`
   - batch CLI entrypoint
   - stable operation registry for contract actions
   - expectation evaluator
3. `Artifact writer`
   - deterministic on-disk failure bundles and event stream
4. `Promotion flow`
   - production failures first land in a captured area
   - human or rule-based review promotes them into the main contract suite

The important boundary is this: the runner operates on domain operations like `sessions.create`, not raw ad hoc HTTP instructions. That keeps the case format stable even if request construction changes later.

## Directory Layout

```text
server/
  app/
    testing/
      __init__.py
      case_loader.py
      models.py
      operation_registry.py
      transport.py
      operations/
        __init__.py
        auth.py
        devices.py
        feedback.py
        profile.py
        sessions.py
        sync.py
      expectations.py
      artifacts.py
      events.py
      runner.py
      redaction.py
      promotion.py
  cases/
    contracts/
      *.json
    fixtures/
      *.json
    captured/
      *.json
    targets.example.json
  scripts/
    run_contract_cases.py
    capture_contract_failure.py
.github/
  workflows/
    contract-cases.yml
```

`server/scripts/run_contract_cases.py` is just the CLI entrypoint. Core logic lives under `server/app/testing/` so the same engine can later be called from CI, local tooling, or a WebSocket control server.

## Case Model

Each case file uses a shared schema:

```json
{
  "case_id": "full_chain_happy_path_v1",
  "title": "Bootstrap + Session Upload + Feedback Happy Path",
  "kind": "full_chain",
  "source": "manual",
  "enabled": true,
  "tags": ["core", "happy_path", "platform"],
  "target_profile": "staging",
  "seed": {
    "user_suffix": "happy01",
    "session_suffix": "happy01"
  },
  "fixtures": {
    "provider": "apple",
    "provider_token_template": "dev:{user_suffix}",
    "device": {
      "client_device_key_template": "case-device-{user_suffix}",
      "platform": "ios",
      "device_name": "Contract Case Device {user_suffix}",
      "app_version": "0.1.0",
      "os_version": "iOS 18.0"
    },
    "session_blob_ref": "fixtures/session_blob_minimal.json",
    "feedback": {
      "predicted_stamina": 61,
      "actual_stamina": 35,
      "label": "fatigued",
      "kss": 8,
      "note": "contract case feedback"
    }
  },
  "steps": [
    {
      "step_id": "create_session",
      "operation": "sessions.create",
      "input": {},
      "expect": {
        "ok": true,
        "status": 200,
        "json_paths": [
          {"path": "$.data.upload.upload_url", "scheme": "https"}
        ]
      },
      "extract": {
        "session_id": "$.data.session.session_id",
        "upload_url": "$.data.upload.upload_url"
      }
    }
  ],
  "artifact_policy": {
    "on_fail": ["http", "domain_payloads", "blob_copy", "step_trace"],
    "on_pass": ["summary"]
  },
  "redaction": {
    "headers": ["authorization"],
    "json_paths": [
      "$.data.access_token",
      "$.data.refresh_token"
    ]
  }
}
```

### Why this schema

- `operation` is domain-level, not transport-level
- `expect` supports contract assertions like `scheme == https`
- `extract` turns one step output into later step inputs
- `fixtures` keeps large payloads out of the main case document
- `source` distinguishes authored cases from captured failures without inventing a second format

## Supported Operations in V1

The initial operation set is intentionally narrow for V1A:

- `auth.sign_up`
- `auth.sign_in`
- `sync.bootstrap`
- `sessions.create`
- `sessions.upload_blob`
- `sessions.finalize`
- `sessions.get`
- `sessions.download`
- `feedback.create`
- `feedback.list`
- `auth.refresh`
- `auth.sign_out`

This is enough to cover the current six-case V1A chain. `devices.list` and `profile.get` remain valid future operations, but they are not required for the first release if the initial case set does not depend on them.

## Runner Architecture

### 1. Case loader

Responsibilities:

- load JSON files from `server/cases/contracts/`
- filter by `case_id`, `tag`, and `enabled`
- resolve fixture references
- expand seed templates
- validate schema before execution

### 2. Run context

Every run gets a stable context object:

- `run_id`
- `target_profile`
- `base_url`
- `target_capabilities`
- `variables`
- `artifacts_dir`
- `event_sink`

This object is the interface between the loader, operations, and artifact writer. It also becomes the stable seam for future WebSocket control.

### 3. Operation registry

Each operation handler:

1. resolves variables from prior extracts
2. builds the real HTTP request
3. sends the request
4. returns a normalized `StepResult`

The runner never embeds per-route request construction in its main loop.

### 3.5 Transport boundary

The runner must depend on a transport abstraction rather than directly hard-coding live HTTP calls in every test path.

V1A has two transport modes:

- fake transport for deterministic unit tests
- live HTTP transport for staging integration runs

This avoids turning every test into a network test and gives the runner a clean seam for retries, failure simulation, and later WebSocket control.

### 4. Expectation evaluator

Supported V1 checks:

- `status`
- envelope `ok`
- JSON path exists
- JSON path equals
- JSON path non-empty
- URL scheme equals `https`
- downloaded blob matches fixture payload

This evaluator is where contract correctness lives. The recent public failure with `http://api.focux.me/...` belongs here as a first-class assertion, not as an accidental smoke script side effect.

### 5. Artifact writer

Artifacts are written under:

```text
artifacts/contracts/<run_id>/
  run.json
  events.jsonl
  cases/
    <case_id>/
      summary.json
      steps/
        01-sign_up.request.json
        01-sign_up.response.json
      payloads/
        session_blob.json
        finalize_payload.json
        feedback_payload.json
```

Failure artifacts are the actual product here. Without them, captured regression is just a noisy log line.

## Event Model

The runner emits a normalized event stream even in batch mode:

- `run_started`
- `case_started`
- `step_started`
- `step_passed`
- `step_failed`
- `artifact_written`
- `case_finished`
- `run_finished`

Each event includes:

- `schema_version`
- `seq`
- `ts`
- `run_id`
- `case_id`
- `step_id`
- `type`
- `status`
- `message`
- `data`

V1 writes these events to `events.jsonl`. V2 can reuse the exact same payloads over WebSocket.

## Capture and Promotion Flow

V1 uses a gated promotion path:

1. production incident is converted into a captured failure bundle
2. redaction runs
3. resulting case lands in `server/cases/captured/`
4. human review or promotion rules decide whether it becomes a durable contract case in `server/cases/contracts/`

This avoids poisoning the main suite with malformed, low-signal, or private data.

V1A must include a minimal capture ingestion command that can take a sanitized failure bundle and emit a captured case document into `server/cases/captured/`.

### Promotion criteria

A captured case can be promoted only if:

- it reproduces against staging
- secrets are redacted
- payload size is reasonable
- it expresses a stable product contract, not a one-off environment glitch

## Replay Target Profiles

V1 introduces named target profiles:

- `staging`
- `local`
- `production` only for read-only debugging, not default replay

The default target profile is `staging`.

The profile abstraction keeps the case documents stable while base URL, auth mode, seed policy, and safety capabilities vary by environment.

Each profile must expose capability flags:

- `allow_writes`
- `allow_capture`
- `allow_shadow_read`

`production` sets `allow_writes = false` by default. A break-glass override is explicit and never the default CLI behavior.

## Initial Case Set

V1 should ship with these cases:

1. `bootstrap_happy_path_v1`
2. `session_upload_happy_path_v1`
3. `feedback_event_happy_path_v1`
4. `full_chain_happy_path_v1`
5. `session_upload_requires_https_url_v1`
6. `feedback_event_without_session_fallback_v1`

Cases 5 and 6 exist because they map directly to bugs and edge behavior already encountered in this repo. They are not optional polish.

## WebSocket Upgrade Path

WebSocket is deferred, not rejected.

The correct future role is `control plane + event streaming`, not replacing the business APIs that already work over HTTPS.

### V2 path

Add:

- `WebSocketControlServer`
- `WebSocketEventSink`

Reuse:

- existing case schema
- existing runner
- existing event payloads
- existing artifact layout

That yields:

- V1: `CLI -> runner -> JSONL + artifacts`
- V2: `WS command -> runner -> JSONL + artifacts + WS events`

No core execution rewrite.

## Risks

1. Case schema turns into a second scripting language.
   - mitigation: fixed `operation` registry, narrow `expect` grammar
2. Captured cases leak secrets or unstable payloads.
   - mitigation: redaction layer and gated promotion
3. Staging drifts too far from production.
   - mitigation: keep authored core cases small and deterministic, periodically validate promoted cases against fresh staging
4. The runner grows into a generic test framework too early.
   - mitigation: V1 only serves platform contract flows already present in the product
5. Artifact growth becomes expensive or noisy.
   - mitigation: V1A defines per-case and per-run byte caps, stores hashes for oversized blobs, and documents cleanup and retention
6. Staging drifts from production without anyone noticing.
   - mitigation: V1A adds a scheduled read-only production-shadow drift check for the core contract path

## Success Criteria

V1A is successful when:

1. a single CLI command can run the core platform contract suite against staging
2. each failure produces a deterministic artifact bundle
3. the recent HTTPS `upload_url` regression is caught by a dedicated case
4. a captured production failure can be transformed into a reviewable case without changing runner code
5. the event model is reused unchanged when WebSocket control is added later
6. the core contract suite runs in CI and fails the build on regression
7. read-only production-shadow checks exist to detect staging drift

## Open Questions Deferred

These are intentionally deferred out of V1:

- automatic screenshot capture from simulator cases
- UI semantic assertions
- zero-touch promotion from production failure to permanent regression

Those belong to the later simulator and visual regression phases.
