<!-- /autoplan restore point: /Users/wujiajun/.gstack/projects/fluxchi/main-autoplan-restore-20260401-163018.md -->
# Contract Case Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build V1A of a staging-first contract regression harness for the FluxChi platform path, covering the core chain with manual core cases, captured failure bundles, CI gating, and a runner/event model that can later back a WebSocket control plane.

**Architecture:** Add a small testing subsystem under `server/app/testing/` that loads declarative contract cases, maps them to explicit platform operations, executes them against a named target profile, and writes structured event and artifact output. Keep transport details inside operation handlers so the case schema stays stable when new replay modes or control planes are added later, and make target profiles enforce capability guards so production is read-only unless a break-glass override is passed.

**Tech Stack:** Python 3, existing server package layout, JSON case files, JSONL event stream, standard library HTTP tooling already used by `server/scripts/smoke_test.py`

---

## File Map

**Create**

- `server/app/testing/__init__.py`
- `server/app/testing/models.py`
- `server/app/testing/case_loader.py`
- `server/app/testing/operation_registry.py`
- `server/app/testing/transport.py`
- `server/app/testing/expectations.py`
- `server/app/testing/events.py`
- `server/app/testing/artifacts.py`
- `server/app/testing/redaction.py`
- `server/app/testing/runner.py`
- `server/app/testing/operations/__init__.py`
- `server/app/testing/operations/auth.py`
- `server/app/testing/operations/feedback.py`
- `server/app/testing/operations/sync.py`
- `server/app/testing/operations/sessions.py`
- `server/cases/targets.example.json`
- `server/scripts/capture_contract_failure.py`
- `server/requirements-dev.txt`
- `server/tests/testing/__init__.py`
- `server/scripts/run_contract_cases.py`
- `server/cases/contracts/bootstrap_happy_path_v1.json`
- `server/cases/contracts/session_upload_happy_path_v1.json`
- `server/cases/contracts/feedback_event_happy_path_v1.json`
- `server/cases/contracts/full_chain_happy_path_v1.json`
- `server/cases/contracts/session_upload_requires_https_url_v1.json`
- `server/cases/contracts/feedback_event_without_session_fallback_v1.json`
- `server/cases/fixtures/session_blob_minimal.json`
- `server/cases/README.md`
- `.github/workflows/contract-cases.yml`

**Modify**

- `server/scripts/smoke_test.py`
- `server/README.md`
- `docs/API-CHANGELOG.md`

**Test**

- use a real `server/tests/testing/` layout in V1A
- split tests into deterministic unit tests and explicit live staging integration runs

---

### Task 0: Bootstrap Test, Target Profile, and Safety Foundations

**Files:**
- Create: `server/requirements-dev.txt`
- Create: `server/tests/testing/__init__.py`
- Create: `server/app/testing/transport.py`
- Create: `server/cases/targets.example.json`
- Test: `server/tests/testing/test_target_profiles.py`

- [ ] **Step 1: Write the failing target profile and transport tests**

Cover:
- `staging` allows mutating operations
- `production` blocks mutating operations by default
- a break-glass override is required for production writes
- fixture and live transports share one interface

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest server/tests/testing/test_target_profiles.py -q
```

Expected:

```text
FAIL ... target profile or transport layer missing
```

- [ ] **Step 3: Add dev test dependencies and base test layout**

Create `server/requirements-dev.txt` with:
- `-r requirements.txt`
- `pytest>=8`

Create `server/tests/testing/__init__.py` so the repo stops pretending a test layout already exists.

- [ ] **Step 4: Implement target profiles and transport abstraction**

Add:
- named target profiles in `server/cases/targets.example.json`
- capability flags including `allow_writes`
- runner preflight hooks that can reject mutating operations on production
- a transport interface with fake and live HTTP implementations

- [ ] **Step 5: Re-run the tests**

Run:

```bash
python3 -m pytest server/tests/testing/test_target_profiles.py -q
```

Expected:

```text
all target profile and transport tests pass
```

- [ ] **Step 6: Commit**

```bash
git add server/requirements-dev.txt server/tests/testing/__init__.py server/app/testing/transport.py server/cases/targets.example.json server/tests/testing/test_target_profiles.py
git commit -m "feat: add contract harness target profiles and transport boundary"
```

### Task 1: Define the Case and Runtime Models

**Files:**
- Create: `server/app/testing/models.py`
- Create: `server/app/testing/__init__.py`
- Test: `server/tests/testing/test_models.py`

- [ ] **Step 1: Write the failing model validation test**

Define tests for:
- case with missing `case_id` should fail validation
- case with unknown `operation` should fail validation
- case with invalid `artifact_policy` should fail validation
- case with valid `sessions.create` step and `scheme=https` expectation should load successfully

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest server/tests/testing/test_models.py -q
```

Expected:

```text
FAIL ... module not found or validation class missing
```

- [ ] **Step 3: Implement the runtime models**

Add explicit dataclasses or pydantic models for:
- `LoadedCase`
- `CaseStep`
- `ExpectationRule`
- `ExtractRule`
- `ArtifactPolicy`
- `RunContext`
- `StepResult`
- `RunEvent`

Model rules:
- `operation` is a string from a fixed registry
- `source` is `manual` or `captured`
- `target_profile` defaults to `staging`
- `artifact_policy.on_fail` is a subset of `http`, `domain_payloads`, `blob_copy`, `step_trace`

- [ ] **Step 4: Re-run the validation test**

Run:

```bash
python3 -m pytest server/tests/testing/test_models.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add server/app/testing/__init__.py server/app/testing/models.py server/tests/testing/test_models.py
git commit -m "feat: add contract case runtime models"
```

### Task 2: Build the Case Loader and Fixture Resolver

**Files:**
- Create: `server/app/testing/case_loader.py`
- Create: `server/cases/fixtures/session_blob_minimal.json`
- Create: `server/cases/contracts/bootstrap_happy_path_v1.json`
- Create: `server/cases/contracts/session_upload_happy_path_v1.json`
- Test: `server/tests/testing/test_case_loader.py`

- [ ] **Step 1: Write the failing loader tests**

Cover:
- fixture reference resolution from `session_blob_ref`
- seed template expansion in provider token and client device key
- filtering by `enabled`
- filtering by `case_id`
- filtering by `tag`

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest server/tests/testing/test_case_loader.py -q
```

Expected:

```text
FAIL ... loader module missing
```

- [ ] **Step 3: Implement the loader**

Add loader behavior:
- read JSON files from `server/cases/contracts/`
- resolve `fixtures/*` references
- expand templates like `{user_suffix}`
- reject malformed files early with clear errors

Seeded fields to support immediately:
- `provider_token_template`
- `client_device_key_template`
- `device_name`

- [ ] **Step 4: Add the first fixture and first two core cases**

Create:
- a minimal session blob fixture that can be uploaded and downloaded deterministically
- a bootstrap happy-path case
- a session upload happy-path case with `upload_url` scheme assertion

- [ ] **Step 5: Re-run the loader tests**

Run:

```bash
python3 -m pytest server/tests/testing/test_case_loader.py -q
```

Expected:

```text
all loader tests pass
```

- [ ] **Step 6: Commit**

```bash
git add server/app/testing/case_loader.py server/cases/fixtures/session_blob_minimal.json server/cases/contracts/bootstrap_happy_path_v1.json server/cases/contracts/session_upload_happy_path_v1.json server/tests/testing/test_case_loader.py
git commit -m "feat: add contract case loader and initial cases"
```

### Task 3: Add the Operation Registry and Core Chain Operations

**Files:**
- Create: `server/app/testing/operation_registry.py`
- Create: `server/app/testing/operations/__init__.py`
- Create: `server/app/testing/operations/auth.py`
- Create: `server/app/testing/operations/feedback.py`
- Create: `server/app/testing/operations/sync.py`
- Create: `server/app/testing/operations/sessions.py`
- Modify: `server/scripts/smoke_test.py`
- Test: `server/tests/testing/test_operation_registry.py`

- [ ] **Step 1: Write failing operation tests**

Cover:
- `auth.sign_up` stores `access_token`, `refresh_token`, and `device_id`
- `auth.refresh` and `auth.sign_out` are represented in the auth handler
- `sync.bootstrap` builds the expected query
- `sessions.create` extracts `session_id`, `upload_url`, and `object_key`
- `sessions.upload_blob` can use the extracted URL
- `sessions.finalize` uses extracted IDs and payload
- `feedback.create` and `feedback.list` work before feedback-dependent cases are introduced

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest server/tests/testing/test_operation_registry.py -q
```

Expected:

```text
FAIL ... registry or operation handlers missing
```

- [ ] **Step 3: Implement the operation registry**

Create a fixed mapping from operation name to handler function.

Handlers must:
- resolve extracted variables from the run context
- return a normalized result object with request, response, parsed JSON, extracted variables, and status
- avoid putting ad hoc request logic in the runner loop

Before implementing handlers, write a case-to-operation matrix in the module docstring or adjacent README that lists:
- each V1A case
- the operations it needs
- the handler file responsible

- [ ] **Step 4: Refactor shared request helpers out of `smoke_test.py` if reuse reduces duplication**

If the runner and smoke script need the same envelope/request helpers, extract the common logic into a shared helper module. Do not duplicate two separate implementations for the same response parsing path.

- [ ] **Step 5: Re-run the operation tests**

Run:

```bash
python3 -m pytest server/tests/testing/test_operation_registry.py -q
```

Expected:

```text
all operation registry tests pass
```

- [ ] **Step 6: Commit**

```bash
git add server/app/testing/operation_registry.py server/app/testing/operations/__init__.py server/app/testing/operations/auth.py server/app/testing/operations/feedback.py server/app/testing/operations/sync.py server/app/testing/operations/sessions.py server/scripts/smoke_test.py server/tests/testing/test_operation_registry.py
git commit -m "feat: add contract operation registry and session handlers"
```

### Task 4: Add Expectation Evaluation and Failure Artifact Writing

**Files:**
- Create: `server/app/testing/expectations.py`
- Create: `server/app/testing/events.py`
- Create: `server/app/testing/artifacts.py`
- Test: `server/tests/testing/test_expectations.py`
- Test: `server/tests/testing/test_artifacts.py`

- [ ] **Step 1: Write failing expectation tests**

Cover:
- `status == 200`
- envelope `ok == true`
- JSON path exists
- JSON path equals
- JSON path non-empty
- URL scheme equals `https`
- downloaded blob content equality

- [ ] **Step 2: Run the expectation test to verify it fails**

Run:

```bash
python3 -m pytest server/tests/testing/test_expectations.py -q
```

Expected:

```text
FAIL ... expectation evaluator missing
```

- [ ] **Step 3: Implement expectation evaluation**

Add explicit evaluators, not a generic `eval`-style expression system.

The evaluator must emit failure messages that identify:
- case ID
- step ID
- rule that failed
- actual vs expected value

Add event fields:
- `schema_version`
- monotonic `seq`

- [ ] **Step 4: Write failing artifact tests**

Cover:
- `events.jsonl` written at run level
- per-step request and response artifacts written under the case directory
- `session_blob.json`, `finalize_payload.json`, and `feedback_payload.json` persisted on failure
- large payloads above the budget are hashed or truncated according to policy

- [ ] **Step 5: Implement event and artifact writers**

Write deterministic artifact paths under:

```text
artifacts/contracts/<run_id>/...
```

Ensure the writer can run in pass and fail modes without branching through duplicated path logic.
Define V1A budgets now:
- max bytes per case artifact set
- max bytes per run artifact set
- cleanup policy for stale run directories

- [ ] **Step 6: Re-run expectation and artifact tests**

Run:

```bash
python3 -m pytest server/tests/testing/test_expectations.py server/tests/testing/test_artifacts.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 7: Commit**

```bash
git add server/app/testing/expectations.py server/app/testing/events.py server/app/testing/artifacts.py server/tests/testing/test_expectations.py server/tests/testing/test_artifacts.py
git commit -m "feat: add contract expectations and artifact writing"
```

### Task 5: Build the Runner and CLI Entry Point

**Files:**
- Create: `server/app/testing/runner.py`
- Create: `server/scripts/run_contract_cases.py`
- Create: `server/cases/contracts/feedback_event_happy_path_v1.json`
- Create: `server/cases/contracts/full_chain_happy_path_v1.json`
- Create: `server/cases/contracts/session_upload_requires_https_url_v1.json`
- Create: `server/cases/contracts/feedback_event_without_session_fallback_v1.json`
- Test: `server/tests/testing/test_runner.py`

- [ ] **Step 1: Write the failing runner tests**

Cover:
- run one case by `--case`
- run multiple cases by `--tag`
- stop a case when a step fails
- emit `step_failed` with artifact references
- write `run_finished` summary

- [ ] **Step 2: Run the runner test to verify it fails**

Run:

```bash
python3 -m pytest server/tests/testing/test_runner.py -q
```

Expected:

```text
FAIL ... runner or CLI missing
```

- [ ] **Step 3: Implement the runner loop**

The loop must:
- load cases
- build a run context
- emit `run_started`
- execute each step in order
- evaluate expectations
- apply extracts into context variables
- write artifacts on failure
- emit `run_finished`
- reject mutating operations when the target profile does not allow writes

- [ ] **Step 4: Add the remaining four initial cases**

Create:
- feedback happy path
- full-chain happy path
- HTTPS upload URL regression case
- feedback fallback without session case

The HTTPS regression case is mandatory. It must fail if the platform returns `http://` upload URLs again.

- [ ] **Step 5: Verify the CLI against staging**

Run:

```bash
python3 -m server.scripts.run_contract_cases --target staging --tag core
```

Expected:

```text
case summaries printed, artifacts directory written, no failed HTTPS scheme check
```

- [ ] **Step 6: Commit**

```bash
git add server/app/testing/runner.py server/scripts/run_contract_cases.py server/cases/contracts/feedback_event_happy_path_v1.json server/cases/contracts/full_chain_happy_path_v1.json server/cases/contracts/session_upload_requires_https_url_v1.json server/cases/contracts/feedback_event_without_session_fallback_v1.json server/tests/testing/test_runner.py
git commit -m "feat: add contract case runner and core suite"
```

### Task 6: Add Redaction, Capture, and Promotion Plumbing

**Files:**
- Create: `server/app/testing/redaction.py`
- Create: `server/app/testing/promotion.py`
- Create: `server/scripts/capture_contract_failure.py`
- Create: `server/cases/README.md`
- Modify: `server/README.md`
- Test: `server/tests/testing/test_redaction.py`
- Test: `server/tests/testing/test_capture_contract_failure.py`

- [ ] **Step 1: Write failing redaction tests**

Cover:
- `Authorization` header redaction
- access and refresh token redaction in envelopes
- stable preservation of replay-relevant IDs and payloads
- signed URL query redaction
- nested secret field redaction

- [ ] **Step 2: Run the redaction test to verify it fails**

Run:

```bash
python3 -m pytest server/tests/testing/test_redaction.py -q
```

Expected:

```text
FAIL ... redaction module missing
```

- [ ] **Step 3: Implement redaction, capture ingestion, and promotion helpers**

Rules:
- redact secrets
- preserve stable contract fields
- scrub signed query params and nested token fields
- write captured failures into `server/cases/captured/`
- keep promotion into `contracts/` as an explicit follow-up action, not automatic

`capture_contract_failure.py` must:
- accept a sanitized failure bundle input
- invoke redaction
- emit a captured case document with stable schema
- refuse malformed inputs

- [ ] **Step 4: Document the workflow**

Document:
- how to run a case
- how to run by tag
- where artifacts land
- how captured cases are reviewed and promoted
- why production is not the default replay target
- the expected incident-to-triage target for captured failures

- [ ] **Step 5: Re-run the redaction tests and one end-to-end sample run**

Run:

```bash
python3 -m pytest server/tests/testing/test_redaction.py server/tests/testing/test_capture_contract_failure.py -q
python3 -m server.scripts.run_contract_cases --target staging --case session_upload_requires_https_url_v1
```

Expected:

```text
redaction tests pass
selected case passes and artifacts are written
```

- [ ] **Step 6: Commit**

```bash
git add server/app/testing/redaction.py server/app/testing/promotion.py server/scripts/capture_contract_failure.py server/cases/README.md server/README.md server/tests/testing/test_redaction.py server/tests/testing/test_capture_contract_failure.py
git commit -m "feat: add contract case redaction and promotion flow"
```

### Task 7: Add CI Gate and Production-Shadow Drift Checks

**Files:**
- Create: `.github/workflows/contract-cases.yml`
- Modify: `server/cases/README.md`
- Modify: `server/README.md`

- [ ] **Step 1: Write the CI and drift-check acceptance criteria**

Document:
- `--tag core` is the release gate
- staging run is blocking
- production-shadow run is read-only and non-blocking
- artifact retention period and cleanup behavior

- [ ] **Step 2: Implement the workflow**

The workflow must:
- install `server/requirements-dev.txt`
- run deterministic unit tests
- run live staging core cases behind explicit environment configuration
- upload artifacts on failure

- [ ] **Step 3: Add a read-only production-shadow command path**

Document a command that:
- uses production target profile
- blocks mutating operations by capability guard
- checks for drift on read-only paths or metadata validation only

- [ ] **Step 4: Verify the workflow configuration**

Run:

```bash
python3 -m server.scripts.run_contract_cases --target staging --tag core
python3 -m server.scripts.run_contract_cases --target production --tag core
```

Expected:

```text
staging run executes normally
production run refuses mutating steps unless an explicit break-glass flag is passed
```

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/contract-cases.yml server/cases/README.md server/README.md
git commit -m "ci: gate core contract cases and add drift checks"
```

### Task 8: Replace Ad Hoc Smoke Reliance With Documented Harness Usage

**Files:**
- Modify: `server/scripts/smoke_test.py`
- Modify: `server/README.md`
- Modify: `docs/API-CHANGELOG.md`

- [ ] **Step 1: Review overlap between `smoke_test.py` and the new harness**

List:
- what stays as a simple smoke script
- what moves to the contract harness
- what duplication should be removed

- [ ] **Step 2: Keep one minimal smoke path and make the harness the primary regression entrypoint**

Document:
- smoke is still useful for quick bring-up
- contract cases are the authoritative regression path for platform chain behavior

- [ ] **Step 3: Verify the documented commands**

Run:

```bash
python3 -m server.scripts.smoke_test
python3 -m server.scripts.run_contract_cases --target staging --tag core
```

Expected:

```text
both commands succeed locally or against the configured profile, with clear differentiation in docs
```

- [ ] **Step 4: Commit**

```bash
git add server/scripts/smoke_test.py server/README.md docs/API-CHANGELOG.md
git commit -m "docs: make contract harness the primary regression path"
```

## Spec Coverage Check

This plan covers the design requirements:

- case registry
- operation registry
- expectation evaluator
- artifact writer
- captured failure promotion path
- staging-first replay
- WebSocket-ready event model without implementing WebSocket now

No simulator automation is included. That is intentionally deferred.

## Placeholder Scan

Red flags intentionally excluded:

- no `TBD`
- no hidden generic "add validation later"
- no generic "write tests" without naming concrete test scopes
- no "similar to previous task" shortcuts

## Type Consistency Check

Core names kept consistent across tasks:

- `LoadedCase`
- `RunContext`
- `StepResult`
- `RunEvent`
- `operation registry`
- `artifact writer`
- `captured -> contracts` promotion flow

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-01-contract-case-harness.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Deferred follow-ups are tracked in `TODOS.md`.

## Review Notes

### Design Review Summary

Skipped, no UI scope. This plan is intentionally limited to platform contract regression, and simulator or screenshot automation remains deferred work.

### CEO Review Summary

**Mode:** Selective expansion

**Premises accepted**

1. Platform contract regression is the right wedge before simulator automation.
2. Batch-first execution is the right V1A sequencing choice.
3. Captured production failures are worth supporting in V1A, but only if capture ingestion is explicitly implemented.
4. Staging-first replay is acceptable only if drift checks exist and production is guarded.

**What already exists**

- a real contract document: `docs/PLATFORM-CONTRACT-V1.md`
- a working public happy-path smoke flow: `server/scripts/smoke_test.py`
- production HTTPS URL handling already proven valuable: `server/app/core/public_url.py`
- iOS client code already exercises the same path through bootstrap, session upload, and feedback events

**NOT in scope**

- simulator UI automation
- screenshot diffing
- WebSocket control plane implementation
- zero-touch captured-case promotion

**Dream state delta**

Current repo state gives you a smoke path. V1A should turn that into a release gate with failure artifacts and captured replay. The 12-month ideal is broader: contract + simulator + visual regression + automated triage. This plan now targets the first durable layer, not the whole future stack.

**CEO findings**

- Critical: V1 must become a real release guardrail, not just a local CLI tool.
- Critical: captured production failures must be implemented, not just promised.
- High: V1A scope must be explicit. Broader operation coverage belongs to V1B.
- High: staging-first replay needs drift controls.
- Medium: WebSocket readiness should stay at schema seam level, not force extra framework now.

**Error & rescue registry**

| Failure | Detection | Rescue |
|--------|--------|--------|
| staging core suite fails in CI | blocking `--tag core` job fails | inspect failure artifacts, rerun selected case against staging, keep release blocked until the regression or environment issue is triaged |
| malformed captured failure bundle | `capture_contract_failure.py` validation rejects input | preserve rejected bundle separately, log validation errors, require sanitized re-export before promotion review |
| production target is invoked with mutating steps | target capability guard rejects the run | abort before network writes, require explicit break-glass flag and operator intent |

**CEO dual voices consensus table**

| Dimension | Claude subagent | Codex | Consensus |
|--------|--------|--------|--------|
| Platform contract before simulator automation | yes | yes | CONFIRMED |
| V1 must be a release guardrail, not local-only tooling | yes | yes | CONFIRMED |
| Captured failure ingestion belongs in V1A | yes | yes | CONFIRMED |
| Staging-first replay needs production-shadow controls | yes | yes | CONFIRMED |
| WebSocket should stay a future seam, not a V1A server surface | yes | yes | CONFIRMED |

**CEO completion summary**

The product review now supports the narrow wedge: ship a real contract gate for the existing platform chain, capture failures in a reusable format, and do not spend V1A on simulator or control-plane machinery.

### Eng Review Summary

**Architecture ASCII diagram**

```text
contract case json/fixture
        |
        v
   case_loader -----> target profile -----> capability guard
        |                    |                    |
        v                    v                    |
     runner -----------> transport ---------------+
        |                 /      \
        |                /        \
        v               v          v
 expectation_eval   fake transport  live HTTP transport
        |
        +----> event sink ----> events.jsonl
        |
        +----> artifact writer ----> artifacts/contracts/<run_id>/
        |
        +----> capture script ----> server/cases/captured/
```

**Test diagram**

| Surface | Coverage Type | Required |
|--------|--------|--------|
| case schema validation | unit | yes |
| fixture resolution and seed expansion | unit | yes |
| target profile safety guard | unit | yes |
| transport abstraction behavior | unit | yes |
| operation registry wiring | unit | yes |
| expectation evaluation | unit | yes |
| artifact writing and caps | unit | yes |
| redaction rules including signed URLs | unit | yes |
| capture ingestion | unit | yes |
| staging core chain replay | live integration | yes |
| production target write refusal | live safety check | yes |

**Test plan artifact**

- `/Users/wujiajun/.gstack/projects/fluxchi/wujiajun-main-test-plan-20260401-170230.md`

**Failure modes registry**

| Failure Mode | Severity | Planned Mitigation |
|--------|--------|--------|
| production replay mutates live data | critical | target capability guard + break-glass flag |
| captured failures leak secrets | high | layered redaction for headers, JSON fields, signed URLs |
| staging drift hides real regressions | high | read-only production-shadow checks |
| artifact volume grows without bound | medium | byte caps, hashing, cleanup policy |
| case schema becomes a mini scripting language | medium | fixed operation registry + narrow expectation grammar |
| live staging failures make all tests flaky | medium | fake transport unit layer + explicit live integration mode |

**Eng findings**

- High: production safety must be enforced, not implied.
- High: capture ingestion is missing from the original implementation plan.
- High: operation coverage must match case set explicitly.
- High: redaction scope must include signed URLs and nested secrets.
- Medium: unit and live integration layers must be separated.
- Medium: event schema needs `schema_version` and monotonic `seq`.
- Medium: artifact budget and retention need explicit V1A limits.

**ENG dual voices consensus table**

| Dimension | Claude subagent | Codex | Consensus |
|--------|--------|--------|--------|
| Architecture sound after target profile and transport split | yes | yes | CONFIRMED |
| Test coverage sufficient for V1A after explicit unit/live split | yes | yes | CONFIRMED |
| Performance risks addressed by artifact caps and staged execution | yes | yes | CONFIRMED |
| Security threats covered by write guards and layered redaction | yes | yes | CONFIRMED |
| Error paths handled by explicit failure modes and capture validation | yes | yes | CONFIRMED |
| Deployment risk manageable with CI gate and production-shadow checks | yes | yes | CONFIRMED |

**Eng completion summary**

The engineering review now describes a buildable system instead of a framework sketch: fixed operation registry, deterministic unit layer, live staging gate, production write refusal, and bounded artifacts.

### Cross-Phase Themes

- The strongest repeated theme is the same one from both review lenses: this must become a real guardrail, not just a framework skeleton.
- The second repeated theme is that staging safety is not enough by itself. Production shadowing and hard write guards are part of the minimum credible version.

## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|----------------|-----------|-----------|----------|
| 1 | CEO | Narrow milestone to V1A core chain only | Mechanical | Explicit over clever | Broad operation catalog and first release plan were inconsistent | Shipping full operation surface now |
| 2 | CEO | Keep captured failure support in V1A | Mechanical | Choose completeness | It is core to the stated product value, not polish | Deferring all capture work to V1B |
| 3 | CEO | Add CI gate to V1A | Mechanical | Bias toward action | Without CI the harness can stay optional and never protect releases | Local CLI only |
| 4 | CEO | Keep WebSocket as future seam only | Taste | Pragmatic | Event schema reuse is enough for now, WS server components are premature | Full WebSocket control plane in V1A |
| 5 | Eng | Add target capability guards | Mechanical | Choose completeness | Production read-only intent must be enforced in code and tests | Relying on docs and operator discipline |
| 6 | Eng | Introduce transport abstraction | Mechanical | Explicit over clever | Unit tests need deterministic transport, live network behavior belongs in integration runs | Network-only test design |
| 7 | Eng | Add signed URL and nested secret redaction | Mechanical | Choose completeness | Captured production failures otherwise risk leaking replay tokens | Header-only redaction |
| 8 | Eng | Add event `schema_version` and `seq` | Mechanical | Explicit over clever | Future WebSocket reuse needs ordered, versioned events | Ad hoc unversioned JSONL events |
| 9 | Eng | Define artifact caps in V1A | Mechanical | Pragmatic | Artifact growth is predictable and cheap to bound now | Deferring retention and budget design |
