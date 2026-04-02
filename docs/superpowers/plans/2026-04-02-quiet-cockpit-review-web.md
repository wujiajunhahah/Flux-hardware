# Quiet Cockpit + Review Board Web Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current FluxChi web dashboard into a quieter live cockpit that defaults to silent logging, exposes deterministic intervention stages, and adds a testable review board for archived sessions and evidence stats.

**Architecture:** Keep the existing FastAPI + static HTML stack, but move new behavior into two focused backend helpers: one deterministic intervention policy for live state updates, and one session review builder for archived recordings. Refactor the front-end into a thin HTML shell plus dedicated CSS/JS assets so the cockpit, intervention feed, evidence cards, and session review board can evolve without another thousand-line HTML file.

**Tech Stack:** Python, FastAPI, pytest, plain HTML/CSS/JS, existing WebSocket/SSE data flow, local session archives, local flywheel SQLite stats.

---

## File Map

### Create

- `src/intervention_policy.py`
  - deterministic silent-log / nudge / escalation stage logic for live payloads
- `src/session_review.py`
  - derive review JSON from archived session packages
- `tests/test_intervention_policy.py`
  - unit tests for intervention stage transitions and cooldown behavior
- `tests/test_session_review.py`
  - unit tests for review summary, drops, and recovery detection
- `web/static/dashboard.css`
  - cockpit + review board styles
- `web/static/dashboard.js`
  - live data rendering, session loading, review board, evidence cards

### Modify

- `web/app.py`
  - wire intervention policy into `processing_loop`
  - expose review endpoint for archived sessions
  - normalize timeline event shapes for the new UI
- `web/static/index.html`
  - convert to shell that loads the new assets and keeps the existing transport / session / camera affordances
- `src/session_insight.py`
  - reuse review analysis for richer stop-session summary text without changing the public contract

## Task 1: Add deterministic intervention policy

**Files:**
- Create: `src/intervention_policy.py`
- Test: `tests/test_intervention_policy.py`

- [ ] **Step 1: Write the failing test for default silent logging**

```python
from src.intervention_policy import InterventionTracker


def make_payload(stamina: float, urgency: float, *, alerts=None):
    return {
        "stamina": {"value": stamina, "state": "fading" if stamina < 60 else "focused"},
        "decision": {
            "urgency": urgency,
            "recommendation": "keep_working" if urgency < 0.5 else "take_break",
            "reasons": ["效率在下降"],
        },
        "fusion": {"alerts": alerts or []},
    }


def test_first_detected_decline_only_creates_silent_log_event():
    tracker = InterventionTracker(sustain_sec=180, nudge_sec=300, cooldown_sec=240)

    state, event = tracker.update(make_payload(54.0, 0.42), now=100.0)

    assert state["stage"] == "silent_log"
    assert event["type"] == "silent_log"
    assert event["should_surface"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_intervention_policy.py::test_first_detected_decline_only_creates_silent_log_event -v`

Expected: FAIL with `ModuleNotFoundError` or missing `InterventionTracker`.

- [ ] **Step 3: Expand the failing tests for escalation behavior**

```python
def test_persistent_decline_escalates_to_light_nudge_after_threshold():
    tracker = InterventionTracker(sustain_sec=180, nudge_sec=300, cooldown_sec=240)

    tracker.update(make_payload(54.0, 0.42), now=100.0)
    state, event = tracker.update(make_payload(46.0, 0.58), now=420.0)

    assert state["stage"] == "light_nudge"
    assert event["type"] == "light_nudge"
    assert event["should_surface"] is True


def test_severe_alert_skips_to_escalation():
    tracker = InterventionTracker(sustain_sec=180, nudge_sec=300, cooldown_sec=240)

    state, event = tracker.update(
        make_payload(18.0, 0.92, alerts=["perclos_severe"]),
        now=100.0,
    )

    assert state["stage"] == "escalation"
    assert event["type"] == "escalation"
    assert "perclos_severe" in state["evidence"]
```

- [ ] **Step 4: Run the full intervention policy test file and confirm all tests fail for the right reason**

Run: `pytest tests/test_intervention_policy.py -v`

Expected: FAIL only because the implementation does not exist yet.

- [ ] **Step 5: Write the minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class InterventionTracker:
    sustain_sec: float = 180.0
    nudge_sec: float = 300.0
    cooldown_sec: float = 240.0
    _decline_started_at: Optional[float] = None
    _last_stage: str = "steady"
    _last_event_at: float = 0.0

    def update(self, payload: Dict[str, Any], now: float) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        stamina = float(((payload.get("stamina") or {}).get("value")) or 0.0)
        decision = payload.get("decision") or {}
        urgency = float(decision.get("urgency") or 0.0)
        alerts = list(((payload.get("fusion") or {}).get("alerts")) or [])

        severe = any(a in {"perclos_severe", "microsleep_detected", "nod_detected"} for a in alerts)
        declining = severe or urgency >= 0.35 or stamina <= 55.0

        if declining and self._decline_started_at is None:
            self._decline_started_at = now
        if not declining:
            self._decline_started_at = None
            self._last_stage = "steady"
            return {
                "stage": "steady",
                "score": urgency,
                "evidence": [],
                "should_surface": False,
            }, None

        active_for = now - (self._decline_started_at or now)
        if severe or urgency >= 0.85 or stamina <= 25.0:
            stage = "escalation"
        elif active_for >= self.nudge_sec or urgency >= 0.55:
            stage = "light_nudge"
        else:
            stage = "silent_log"

        state = {
            "stage": stage,
            "score": round(max(urgency, 1.0 - stamina / 100.0), 3),
            "evidence": alerts or list(decision.get("reasons") or []),
            "should_surface": stage in {"light_nudge", "escalation"},
        }

        event = None
        if stage != self._last_stage and (now - self._last_event_at >= self.cooldown_sec or stage == "escalation"):
            event = {"type": stage, **state, "time": now}
            self._last_event_at = now
            self._last_stage = stage

        return state, event
```

- [ ] **Step 6: Run the intervention tests to verify they pass**

Run: `pytest tests/test_intervention_policy.py -v`

Expected: PASS with 3 passing tests.

- [ ] **Step 7: Commit**

```bash
git add tests/test_intervention_policy.py src/intervention_policy.py
git commit -m "feat: add deterministic intervention policy"
```

## Task 2: Add session review builder and richer stop-session summary

**Files:**
- Create: `src/session_review.py`
- Modify: `src/session_insight.py`
- Test: `tests/test_session_review.py`

- [ ] **Step 1: Write the failing review tests**

```python
from src.session_review import build_session_review


def make_pkg(staminas, tensions=None):
    tensions = tensions or [0.2] * len(staminas)
    return {
        "session": {"id": "abc12345", "durationSec": 900},
        "snapshots": [
            {
                "t": idx * 60,
                "stamina": stamina,
                "ten": tensions[idx],
                "fat": 0.2,
                "con": 0.7,
                "state": "focused" if stamina >= 60 else "fading",
                "act": "finger_movement",
            }
            for idx, stamina in enumerate(staminas)
        ],
    }


def test_review_builder_reports_largest_drop_and_recovery_window():
    review = build_session_review(make_pkg([82, 79, 74, 51, 49, 63, 69]))

    assert review["summary"]["largest_drop"] == 33
    assert review["summary"]["recovered"] is True
    assert review["moments"]["lowest"]["stamina"] == 49


def test_review_builder_flags_high_tension_segments():
    review = build_session_review(make_pkg([78, 75, 69, 64], tensions=[0.2, 0.62, 0.67, 0.58]))

    assert review["flags"][0]["type"] == "high_tension"
    assert review["flags"][0]["minutes"] >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_session_review.py -v`

Expected: FAIL because `build_session_review` does not exist yet.

- [ ] **Step 3: Write the minimal review implementation**

```python
from __future__ import annotations

from typing import Any, Dict, List


def build_session_review(pkg: Dict[str, Any]) -> Dict[str, Any]:
    snaps: List[Dict[str, Any]] = list(pkg.get("snapshots") or [])
    if not snaps:
        return {
            "summary": {"largest_drop": 0, "recovered": False},
            "moments": {},
            "flags": [],
            "series": [],
        }

    staminas = [float(s.get("stamina", 0) or 0) for s in snaps]
    tensions = [float(s.get("ten", 0) or 0) for s in snaps]
    largest_drop = round(max(staminas) - min(staminas))
    low_idx = min(range(len(staminas)), key=staminas.__getitem__)
    recovered = low_idx < len(staminas) - 1 and max(staminas[low_idx + 1:]) - staminas[low_idx] >= 10

    flags = []
    high_tension_count = sum(1 for value in tensions if value >= 0.55)
    if high_tension_count:
        flags.append({"type": "high_tension", "minutes": high_tension_count})

    return {
        "summary": {
            "average_stamina": round(sum(staminas) / len(staminas), 1),
            "largest_drop": largest_drop,
            "recovered": recovered,
            "duration_min": round(float(((pkg.get("session") or {}).get("durationSec")) or 0) / 60.0, 1),
        },
        "moments": {
            "lowest": {"index": low_idx, "stamina": round(staminas[low_idx])},
            "highest": {"index": max(range(len(staminas)), key=staminas.__getitem__), "stamina": round(max(staminas))},
        },
        "flags": flags,
        "series": [
            {
                "minute": round(float(s.get("t", 0) or 0) / 60.0, 1),
                "stamina": float(s.get("stamina", 0) or 0),
                "tension": float(s.get("ten", 0) or 0),
                "fatigue": float(s.get("fat", 0) or 0),
            }
            for s in snaps
        ],
    }
```

- [ ] **Step 4: Update `src/session_insight.py` to reuse the review output**

```python
from .session_review import build_session_review


def session_insight_text(pkg: Dict[str, Any]) -> str:
    review = build_session_review(pkg)
    summary = review["summary"]
    if not review["series"]:
        return "暂无快照，无法生成洞察。"
    parts = [
        f"时长约 {summary['duration_min']:.1f} 分钟。",
        f"平均耐力 {summary['average_stamina']:.0f}。",
        f"最大下滑 {summary['largest_drop']:.0f}。",
    ]
    parts.append("后段出现恢复。" if summary["recovered"] else "后段恢复不明显。")
    if review["flags"]:
        parts.append("张力段偏高，建议检查姿势与休息节奏。")
    return " ".join(parts)
```

- [ ] **Step 5: Run review tests**

Run: `pytest tests/test_session_review.py -v`

Expected: PASS with 2 passing tests.

- [ ] **Step 6: Commit**

```bash
git add tests/test_session_review.py src/session_review.py src/session_insight.py
git commit -m "feat: add session review builder"
```

## Task 3: Wire policy and review into FastAPI

**Files:**
- Modify: `web/app.py`
- Test: `tests/test_intervention_policy.py`, `tests/test_session_review.py`

- [ ] **Step 1: Write the failing API test for archived session review**

```python
import json
from fastapi.testclient import TestClient

import web.app as web_app


def test_session_review_endpoint_returns_review_payload(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr("src.session_store.SESSIONS_DIR", session_dir)

    pkg = {
        "session": {"id": "12345678", "durationSec": 600},
        "snapshots": [{"t": 0, "stamina": 80, "ten": 0.2, "fat": 0.1, "con": 0.7}],
    }
    (session_dir / "fluxchi_20260402_120000_12345678.json").write_text(json.dumps(pkg), encoding="utf-8")

    with TestClient(web_app.app) as client:
        resp = client.get("/api/v1/sessions/12345678/review")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["data"]["summary"]["average_stamina"] == 80.0
```

- [ ] **Step 2: Run the API test to verify it fails**

Run: `pytest tests/test_session_review.py::test_session_review_endpoint_returns_review_payload -v`

Expected: FAIL with 404 or missing review endpoint.

- [ ] **Step 3: Modify `web/app.py` to attach intervention state and expose review**

```python
from src.intervention_policy import InterventionTracker
from src.session_review import build_session_review

intervention_tracker: Optional[InterventionTracker] = None


@app.get("/api/v1/sessions/{session_id}/review", tags=["Sessions"], summary="读取结构化复盘")
async def v1_sessions_review(session_id: str):
    try:
        pkg = load_session(session_id)
    except ValueError:
        return _err("bad_request", "非法 session_id")
    except FileNotFoundError:
        return _err("not_found", "会话不存在")
    return _ok(build_session_review(pkg))


async def processing_loop():
    global intervention_tracker
    intervention_tracker = InterventionTracker()
    ...
    intervention_state, intervention_event = intervention_tracker.update(payload, now=now)
    payload["intervention"] = intervention_state
    if intervention_event:
        timeline.append(intervention_event)
```

- [ ] **Step 4: Normalize timeline trimming for the new event shapes**

```python
if intervention_event:
    timeline.append(intervention_event)
    if len(timeline) > 200:
        timeline[:] = timeline[-100:]
```

- [ ] **Step 5: Run the backend tests**

Run: `pytest tests/test_intervention_policy.py tests/test_session_review.py -v`

Expected: PASS with all tests green, including the new endpoint test.

- [ ] **Step 6: Commit**

```bash
git add web/app.py tests/test_intervention_policy.py tests/test_session_review.py
git commit -m "feat: expose intervention and review data"
```

## Task 4: Build the quieter cockpit and review board UI

**Files:**
- Create: `web/static/dashboard.css`
- Create: `web/static/dashboard.js`
- Modify: `web/static/index.html`

- [ ] **Step 1: Replace the monolithic HTML body with a stable shell**

```html
<body>
  <div id="app" class="app-shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">FluxChi</p>
        <h1>Quiet Cockpit</h1>
      </div>
      <div id="liveStage" class="stage-pill">steady</div>
    </header>

    <main class="layout">
      <section class="panel hero-panel">
        <div class="hero-score" id="heroScore">--</div>
        <p id="heroState">Waiting for signal</p>
        <p id="heroReason">Silent by default. Evidence first.</p>
      </section>

      <section class="panel protocol-panel">
        <h2>Intervention</h2>
        <div id="protocolRail"></div>
        <div id="eventFeed" class="event-feed"></div>
      </section>

      <section class="panel evidence-panel">
        <h2>Evidence</h2>
        <div id="evidenceCards" class="evidence-grid"></div>
      </section>

      <section class="panel review-list-panel">
        <h2>Recent Sessions</h2>
        <div id="sessionList" class="session-list"></div>
      </section>

      <section class="panel review-board-panel">
        <h2>Review Board</h2>
        <div id="reviewBoard">No session selected yet.</div>
      </section>
    </main>
  </div>
  <script type="module" src="/static/dashboard.js"></script>
</body>
```

- [ ] **Step 2: Add the minimal stylesheet for the new layout**

```css
:root {
  --bg: #07111a;
  --panel: rgba(10, 18, 28, 0.92);
  --line: rgba(148, 163, 184, 0.14);
  --text: #e5eef8;
  --muted: #9aa9ba;
  --accent: #8cf0b5;
  --warn: #f7c65c;
  --danger: #ff6b6b;
}

body {
  margin: 0;
  font-family: "JetBrains Mono", monospace;
  background:
    radial-gradient(circle at top, rgba(76, 201, 240, 0.18), transparent 32%),
    linear-gradient(180deg, #061018 0%, #04070c 100%);
  color: var(--text);
}

.layout {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 18px;
  padding: 18px;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 18px;
  backdrop-filter: blur(18px);
}
```

- [ ] **Step 3: Implement the dashboard client**

```js
const state = {
  sessions: [],
  selectedSessionId: null,
  timeline: [],
};

async function loadSessions() {
  const resp = await fetch("/api/v1/sessions");
  const body = await resp.json();
  state.sessions = body.ok ? body.data.sessions : [];
  renderSessionList();
  if (!state.selectedSessionId && state.sessions.length) {
    state.selectedSessionId = state.sessions[0].id;
    await loadReview(state.selectedSessionId);
  }
}

async function loadReview(sessionId) {
  const resp = await fetch(`/api/v1/sessions/${sessionId}/review`);
  const body = await resp.json();
  renderReviewBoard(body.ok ? body.data : null);
}

async function loadEvidence() {
  const [sessionsResp, flywheelResp] = await Promise.all([
    fetch("/api/v1/sessions"),
    fetch("/api/v1/flywheel/stats"),
  ]);
  ...
}

function connectLive() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onmessage = ({ data }) => {
    const payload = JSON.parse(data);
    if (payload.type === "state_update") renderLive(payload);
  };
}
```

- [ ] **Step 4: Keep the existing transport, session record, vision, and label controls inside the new shell**

```js
// Re-bind existing endpoints:
// /api/v1/transport
// /api/v1/transport/apply
// /api/v1/sessions/web/start
// /api/v1/sessions/web/stop
// /api/v1/flywheel/label
// /static/vision.html
```

- [ ] **Step 5: Manual UI verification**

Run:

```bash
python web/app.py --demo --web-port 18080
```

Then verify:

```bash
curl -s http://127.0.0.1:18080/api/v1/state | jq '.data.intervention'
curl -s http://127.0.0.1:18080/api/v1/sessions | jq '.data.sessions[0]'
```

Expected:

- live payload contains `intervention`
- recent sessions load without JS errors
- review board renders from archived session review data

- [ ] **Step 6: Commit**

```bash
git add web/static/index.html web/static/dashboard.css web/static/dashboard.js
git commit -m "feat: add quiet cockpit review UI"
```

## Task 5: Final verification

**Files:**
- Verify: `tests/test_intervention_policy.py`
- Verify: `tests/test_session_review.py`
- Verify: `web/app.py`
- Verify: `web/static/index.html`
- Verify: `web/static/dashboard.css`
- Verify: `web/static/dashboard.js`

- [ ] **Step 1: Run the focused automated checks**

Run:

```bash
pytest tests/test_intervention_policy.py tests/test_session_review.py -v
```

Expected: PASS with zero failures.

- [ ] **Step 2: Run the live demo server**

Run:

```bash
python web/app.py --demo --web-port 18080
```

Expected: server starts, demo data streams, no import errors.

- [ ] **Step 3: Dogfood the UI with the browser workflow**

Run with the browse tool:

```bash
$B goto http://127.0.0.1:18080
$B snapshot -i -a -o /tmp/fluxchi-cockpit.png
$B console
$B js "document.querySelector('#reviewBoard')?.textContent"
```

Expected:

- no blocking console errors
- cockpit shell visible
- review board present

- [ ] **Step 4: Capture final evidence**

Run:

```bash
curl -s http://127.0.0.1:18080/api/v1/state | jq '.data.intervention'
curl -s http://127.0.0.1:18080/api/v1/timeline | jq '.data.events[:5]'
curl -s http://127.0.0.1:18080/api/v1/flywheel/stats | jq '.data'
```

Expected:

- intervention stage is present
- timeline events use `silent_log` / `light_nudge` / `escalation`
- evidence stats are readable by the UI

- [ ] **Step 5: Final commit**

```bash
git add src/intervention_policy.py src/session_review.py src/session_insight.py web/app.py web/static/index.html web/static/dashboard.css web/static/dashboard.js tests/test_intervention_policy.py tests/test_session_review.py
git commit -m "feat: ship quiet cockpit review workflow"
```

## Self-Review

### Spec coverage

- quiet, low-friction live experience: covered by intervention policy + cockpit UI
- default silent logging before visible nudges: covered by Task 1 and Task 3
- review board after sessions: covered by Task 2 and Task 4
- evidence/data flywheel support: covered by Task 2, Task 4, and Task 5

### Placeholder scan

- no `TODO` / `TBD`
- each task has exact files
- each test step has explicit commands

### Type consistency

- `InterventionTracker.update()` returns `(state, event)`
- `build_session_review()` returns `summary`, `moments`, `flags`, `series`
- `payload["intervention"]` is the live UI contract
