# Platform Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one script entrypoint that runs the shared backend platform checks for staging core and optional production shadow validation.

**Architecture:** Build a tiny orchestration layer around the existing contract harness and shadow target preparation code. Keep orchestration logic in one focused testing module and expose a thin CLI wrapper for local, server, and CI use.

**Tech Stack:** Python 3, existing `server/app/testing/` helpers, argparse, pytest

---

### Task 1: Add Orchestration Tests First

**Files:**
- Create: `server/tests/testing/test_platform_checks.py`

- [ ] **Step 1: Write failing tests for staging-only and shadow flows**
- [ ] **Step 2: Run the tests to verify they fail because the orchestration module does not exist**

### Task 2: Implement Platform Check Orchestration

**Files:**
- Create: `server/app/testing/platform_checks.py`

- [ ] **Step 1: Implement a pure orchestration function that can call injected `run_cases` and shadow prep helpers**
- [ ] **Step 2: Return one aggregated summary object with per-check entries and top-level `ok`**
- [ ] **Step 3: Re-run the orchestration tests**

### Task 3: Add CLI Entry Point

**Files:**
- Create: `server/scripts/run_platform_checks.py`

- [ ] **Step 1: Add CLI parsing for staging-core and optional production shadow execution**
- [ ] **Step 2: Fail fast when shadow prep is requested without the required provider token**
- [ ] **Step 3: Add CLI-focused tests or direct function tests for flag handling**

### Task 4: Document and Verify

**Files:**
- Modify: `server/README.md`
- Modify: `server/cases/README.md`
- Modify: `docs/API-CHANGELOG.md`

- [ ] **Step 1: Document the new unified platform check command**
- [ ] **Step 2: Run targeted tests and the full `server/tests/testing` suite**
- [ ] **Step 3: Run `python3 -m server.scripts.run_platform_checks --help`**
