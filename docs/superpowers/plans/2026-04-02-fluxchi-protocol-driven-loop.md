# FluxChi Protocol-Driven Quiet Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a protocol-driven middle layer that turns FluxChi's quiet intervention logic, review board, post-session summary, and next-hypothesis generation into one versioned deterministic system.

**Architecture:** Introduce a versioned YAML protocol pack under `config/fluxchi/protocols`, load it through typed Python models, and interpret it through a shared protocol runtime for both live intervention decisions and archived session review. Keep the existing FastAPI + static HTML stack, but move product rule ownership out of scattered Python and JavaScript constants into the protocol pack and backend runtime.

**Tech Stack:** Python, FastAPI, PyYAML, pytest, plain HTML/CSS/JS, existing session archive flow, existing in-memory live timeline.

---

## File Map

### Create

- `config/fluxchi/protocols/quiet_loop_v1.yaml`
  - first deterministic protocol pack for live intervention, review cards, summary blocks, and hypothesis templates
- `src/protocol_models.py`
  - typed dataclasses for protocol pack sections
- `src/protocol_loader.py`
  - load and validate protocol YAML into typed models
- `src/protocol_runtime.py`
  - deterministic interpreter for live intervention stages and protocol-aware review cards
- `src/protocol_summary.py`
  - build structured summary blocks and next-hypothesis objects from review output
- `src/protocol_evidence.py`
  - normalize live protocol events and archived effect-evidence records
- `tests/test_protocol_loader.py`
  - protocol loading and validation tests
- `tests/test_protocol_runtime.py`
  - live stage resolution and review-card generation tests
- `tests/test_protocol_summary.py`
  - summary plan and next-hypothesis tests
- `tests/test_protocol_api.py`
  - FastAPI response-shape tests for protocol fields
- `web/static/protocol-ui.js`
  - protocol-driven render helpers for live stage, review cards, and summary blocks

### Modify

- `src/intervention_policy.py`
  - keep decline timing state, but delegate stage resolution and copy to protocol runtime
- `src/session_insight.py`
  - build short Chinese insight text from structured protocol summary blocks
- `web/app.py`
  - load the default protocol pack once, attach protocol metadata to live payloads, add review/summary/hypothesis to session review responses, and archive protocol evidence with web sessions
- `web/static/dashboard.js`
  - remove hard-coded protocol copy and render from backend protocol payloads
- `web/static/dashboard.css`
  - add styles for protocol cards, summary blocks, and hypothesis sections
- `web/static/index.html`
  - add slots for structured summary and next-experiment content
- `tests/test_session_review.py`
  - keep existing coverage, but assert the expanded review endpoint contract

## Task 1: Add the versioned protocol pack and loader

**Files:**
- Create: `config/fluxchi/protocols/quiet_loop_v1.yaml`
- Create: `src/protocol_models.py`
- Create: `src/protocol_loader.py`
- Test: `tests/test_protocol_loader.py`

- [ ] **Step 1: Write the failing loader tests**

```python
from pathlib import Path

import pytest

from src.protocol_loader import DEFAULT_PROTOCOL_PATH, load_protocol_pack


def test_load_default_protocol_pack_exposes_expected_stage_order():
    pack = load_protocol_pack()

    assert pack.protocol_id == "quiet_loop_v1"
    assert [stage.id for stage in pack.intervention.stages] == [
        "silent_log",
        "light_nudge",
        "escalation",
    ]
    assert pack.review.cards[0].id == "largest_drop"


def test_loader_rejects_protocol_without_intervention_stages(tmp_path: Path):
    bad_path = tmp_path / "broken.yaml"
    bad_path.write_text(
        """
protocol_id: broken
intervention:
  stages: []
review:
  cards: []
summary:
  blocks: []
hypothesis:
  templates: []
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one intervention stage"):
        load_protocol_pack(bad_path)
```

- [ ] **Step 2: Run the loader test file to verify it fails**

Run: `pytest tests/test_protocol_loader.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.protocol_loader'`.

- [ ] **Step 3: Write the minimal protocol pack and loader implementation**

```yaml
# config/fluxchi/protocols/quiet_loop_v1.yaml
protocol_id: quiet_loop_v1
label: FluxChi Quiet Loop v1

intervention:
  stages:
    - id: silent_log
      title: Silent Logging
      reason_copy: 状态开始下滑，先静默记录，不打断工作。
      surface: silent
      next_step: keep_logging
      enters_when:
        any:
          - urgency_gte: 0.35
          - stamina_lte: 55

    - id: light_nudge
      title: Gentle Nudge
      reason_copy: 状态持续走低，已到最轻提醒阈值。
      surface: visible
      next_step: nudge_once
      enters_when:
        any:
          - sustained_decline_sec_gte: 300
          - urgency_gte: 0.55

    - id: escalation
      title: Escalation
      reason_copy: 出现强风险信号，建议暂停并进入复盘。
      surface: visible
      next_step: pause_and_review
      enters_when:
        any:
          - alert_in: [perclos_severe, microsleep_detected, nod_detected]
          - urgency_gte: 0.85
          - stamina_lte: 25

review:
  cards:
    - id: largest_drop
      title: 明显下滑窗口
      severity: warn
      if:
        largest_drop_gte: 18
      body: 会话中出现明显下滑，值得回看下滑前后的活动和张力变化。
      suggests_experiment: false

    - id: high_tension_segment
      title: 张力偏高片段
      severity: warn
      if:
        flag_type: high_tension
      body: 张力段持续偏高，可能和姿势、用力方式或节奏有关。
      suggests_experiment: true

    - id: failed_recovery
      title: 恢复不足
      severity: danger
      if:
        recovered_is: false
        largest_drop_gte: 20
      body: 明显下滑后没有出现足够恢复，适合优先设计下一次干预实验。
      suggests_experiment: true

summary:
  blocks:
    - id: observation
      label: Observation
    - id: interpretation
      label: Likely Interpretation
    - id: next_experiment
      label: Next Experiment
    - id: confidence
      label: Confidence

hypothesis:
  templates:
    - id: high_tension_recovery
      when_card: high_tension_segment
      hypothesis: 当张力持续偏高超过 2 分钟时，耐力下滑更容易发生。
      next_experiment: 下次连续 2 个 session，在张力升高时尝试 30 秒肩颈重置。
      confidence: medium
```

```python
# src/protocol_models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class StageSpec:
    id: str
    title: str
    reason_copy: str
    surface: str
    next_step: str
    enters_when: Dict[str, Any]


@dataclass(frozen=True)
class InterventionSpec:
    stages: List[StageSpec]


@dataclass(frozen=True)
class ReviewCardSpec:
    id: str
    title: str
    severity: str
    condition: Dict[str, Any]
    body: str
    suggests_experiment: bool = False


@dataclass(frozen=True)
class ReviewSpec:
    cards: List[ReviewCardSpec]


@dataclass(frozen=True)
class SummaryBlockSpec:
    id: str
    label: str


@dataclass(frozen=True)
class SummarySpec:
    blocks: List[SummaryBlockSpec]


@dataclass(frozen=True)
class HypothesisTemplateSpec:
    id: str
    when_card: str
    hypothesis: str
    next_experiment: str
    confidence: str


@dataclass(frozen=True)
class HypothesisSpec:
    templates: List[HypothesisTemplateSpec]


@dataclass(frozen=True)
class ProtocolPack:
    protocol_id: str
    label: str
    intervention: InterventionSpec
    review: ReviewSpec
    summary: SummarySpec
    hypothesis: HypothesisSpec
```

```python
# src/protocol_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .protocol_models import (
    HypothesisSpec,
    HypothesisTemplateSpec,
    InterventionSpec,
    ProtocolPack,
    ReviewCardSpec,
    ReviewSpec,
    StageSpec,
    SummaryBlockSpec,
    SummarySpec,
)


DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "fluxchi"
    / "protocols"
    / "quiet_loop_v1.yaml"
)


def _require_list(value: Any, field_name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


def load_protocol_pack(path: Path | str | None = None) -> ProtocolPack:
    protocol_path = Path(path) if path else DEFAULT_PROTOCOL_PATH
    raw = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}

    stages_raw = _require_list(((raw.get("intervention") or {}).get("stages")), "intervention.stages")
    if not stages_raw:
        raise ValueError("protocol must define at least one intervention stage")

    review_cards_raw = _require_list(((raw.get("review") or {}).get("cards")), "review.cards")
    summary_blocks_raw = _require_list(((raw.get("summary") or {}).get("blocks")), "summary.blocks")
    hypothesis_raw = _require_list(((raw.get("hypothesis") or {}).get("templates")), "hypothesis.templates")

    return ProtocolPack(
        protocol_id=str(raw.get("protocol_id") or "").strip(),
        label=str(raw.get("label") or "").strip(),
        intervention=InterventionSpec(
            stages=[
                StageSpec(
                    id=str(item["id"]),
                    title=str(item["title"]),
                    reason_copy=str(item["reason_copy"]),
                    surface=str(item["surface"]),
                    next_step=str(item["next_step"]),
                    enters_when=dict(item.get("enters_when") or {}),
                )
                for item in stages_raw
            ]
        ),
        review=ReviewSpec(
            cards=[
                ReviewCardSpec(
                    id=str(item["id"]),
                    title=str(item["title"]),
                    severity=str(item["severity"]),
                    condition=dict(item.get("if") or {}),
                    body=str(item["body"]),
                    suggests_experiment=bool(item.get("suggests_experiment", False)),
                )
                for item in review_cards_raw
            ]
        ),
        summary=SummarySpec(
            blocks=[
                SummaryBlockSpec(id=str(item["id"]), label=str(item["label"]))
                for item in summary_blocks_raw
            ]
        ),
        hypothesis=HypothesisSpec(
            templates=[
                HypothesisTemplateSpec(
                    id=str(item["id"]),
                    when_card=str(item["when_card"]),
                    hypothesis=str(item["hypothesis"]),
                    next_experiment=str(item["next_experiment"]),
                    confidence=str(item["confidence"]),
                )
                for item in hypothesis_raw
            ]
        ),
    )
```

- [ ] **Step 4: Run the loader tests to verify they pass**

Run: `pytest tests/test_protocol_loader.py -v`

Expected: PASS with 2 passing tests.

- [ ] **Step 5: Commit**

```bash
git add config/fluxchi/protocols/quiet_loop_v1.yaml src/protocol_models.py src/protocol_loader.py tests/test_protocol_loader.py
git commit -m "feat: add FluxChi protocol pack loader"
```

## Task 2: Add live protocol runtime and refactor intervention tracking

**Files:**
- Create: `src/protocol_runtime.py`
- Modify: `src/intervention_policy.py`
- Test: `tests/test_protocol_runtime.py`

- [ ] **Step 1: Write the failing live-runtime tests**

```python
from src.protocol_loader import load_protocol_pack
from src.protocol_runtime import ProtocolRuntime
from src.intervention_policy import InterventionTracker


def make_payload(stamina: float, urgency: float, *, alerts=None, reasons=None):
    return {
        "stamina": {"value": stamina, "state": "fading" if stamina < 60 else "focused"},
        "decision": {
            "urgency": urgency,
            "reasons": reasons or ["效率在下降"],
        },
        "fusion": {"alerts": alerts or []},
    }


def test_protocol_runtime_resolves_light_nudge_with_rule_metadata():
    runtime = ProtocolRuntime(load_protocol_pack())

    decision = runtime.evaluate_live(
        make_payload(47.0, 0.58),
        active_for_sec=360.0,
        previous_stage="silent_log",
    )

    assert decision["stage"] == "light_nudge"
    assert decision["protocol"]["id"] == "quiet_loop_v1"
    assert decision["protocol"]["rule_id"] == "light_nudge"
    assert decision["protocol"]["surface"] == "visible"


def test_intervention_tracker_embeds_protocol_metadata_in_state():
    tracker = InterventionTracker(sustain_sec=180, nudge_sec=300, cooldown_sec=240)

    tracker.update(make_payload(54.0, 0.42), now=100.0)
    state, event = tracker.update(make_payload(47.0, 0.58), now=420.0)

    assert state["stage"] == "light_nudge"
    assert state["protocol"]["rule_id"] == "light_nudge"
    assert event["protocol"]["surface"] == "visible"
```

- [ ] **Step 2: Run the live-runtime tests to verify they fail**

Run: `pytest tests/test_protocol_runtime.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.protocol_runtime'`.

- [ ] **Step 3: Write the minimal live protocol runtime and tracker integration**

```python
# src/protocol_runtime.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .protocol_models import ProtocolPack, ReviewCardSpec, StageSpec


SEVERE_ALERTS = {"perclos_severe", "microsleep_detected", "nod_detected"}


class ProtocolRuntime:
    def __init__(self, pack: ProtocolPack):
        self.pack = pack

    def evaluate_live(
        self,
        payload: Dict[str, Any],
        *,
        active_for_sec: float,
        previous_stage: str,
    ) -> Dict[str, Any]:
        stamina = float(((payload.get("stamina") or {}).get("value")) or 0.0)
        decision = payload.get("decision") or {}
        urgency = float(decision.get("urgency") or 0.0)
        reasons = list(decision.get("reasons") or [])
        alerts = list(((payload.get("fusion") or {}).get("alerts")) or [])

        matched_stage = None
        for stage in reversed(self.pack.intervention.stages):
            if self._matches_live_stage(stage, stamina=stamina, urgency=urgency, alerts=alerts, active_for_sec=active_for_sec):
                matched_stage = stage
                break

        if matched_stage is None:
            return {
                "stage": "steady",
                "score": 0.0,
                "active_for_sec": 0.0,
                "evidence": [],
                "should_surface": False,
                "next_step": "keep_sensing",
                "protocol": {
                    "id": self.pack.protocol_id,
                    "rule_id": "steady",
                    "surface": "silent",
                    "title": "Quiet Monitoring",
                    "reason_copy": "Silent by default. Evidence first.",
                },
            }

        score = round(max(urgency, max(0.0, 1.0 - stamina / 100.0)), 3)
        evidence = alerts or reasons
        return {
            "stage": matched_stage.id,
            "score": score,
            "active_for_sec": round(active_for_sec, 1),
            "evidence": evidence,
            "should_surface": matched_stage.surface == "visible",
            "next_step": matched_stage.next_step,
            "protocol": {
                "id": self.pack.protocol_id,
                "rule_id": matched_stage.id,
                "surface": matched_stage.surface,
                "title": matched_stage.title,
                "reason_copy": matched_stage.reason_copy,
                "previous_stage": previous_stage,
            },
        }

    def _matches_live_stage(
        self,
        stage: StageSpec,
        *,
        stamina: float,
        urgency: float,
        alerts: Iterable[str],
        active_for_sec: float,
    ) -> bool:
        conditions = list((stage.enters_when or {}).get("any") or [])
        for condition in conditions:
            if "urgency_gte" in condition and urgency >= float(condition["urgency_gte"]):
                return True
            if "stamina_lte" in condition and stamina <= float(condition["stamina_lte"]):
                return True
            if "sustained_decline_sec_gte" in condition and active_for_sec >= float(condition["sustained_decline_sec_gte"]):
                return True
            if "alert_in" in condition and any(alert in set(condition["alert_in"]) for alert in alerts):
                return True
        return False
```

```python
# src/intervention_policy.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .protocol_loader import load_protocol_pack
from .protocol_runtime import ProtocolRuntime


@dataclass
class InterventionTracker:
    sustain_sec: float = 180.0
    nudge_sec: float = 300.0
    cooldown_sec: float = 240.0
    runtime: ProtocolRuntime = field(default_factory=lambda: ProtocolRuntime(load_protocol_pack()))
    _decline_started_at: Optional[float] = None
    _last_stage: str = "steady"
    _last_event_at: float = 0.0

    def update(self, payload: Dict[str, Any], now: float) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        stamina = float((((payload.get("stamina") or {}).get("value")) or 0.0))
        urgency = float((((payload.get("decision") or {}).get("urgency")) or 0.0))
        alerts = list((((payload.get("fusion") or {}).get("alerts")) or []))
        declining = bool(alerts) or urgency >= 0.35 or stamina <= 55.0

        if declining and self._decline_started_at is None:
            self._decline_started_at = now
        if not declining:
            previous_stage = self._last_stage
            self._decline_started_at = None
            self._last_stage = "steady"
            state = self.runtime.evaluate_live(payload, active_for_sec=0.0, previous_stage=previous_stage)
            if previous_stage in {"silent_log", "light_nudge", "escalation"}:
                return {"stage": "steady", **state}, {"type": "recovered", **state, "time": now}
            return {"stage": "steady", **state}, None

        active_for_sec = now - (self._decline_started_at or now)
        state = self.runtime.evaluate_live(payload, active_for_sec=active_for_sec, previous_stage=self._last_stage)

        event = None
        should_emit = state["stage"] != self._last_stage and (
            self._last_event_at == 0.0
            or state["stage"] == "escalation"
            or (now - self._last_event_at) >= self.cooldown_sec
        )
        if should_emit:
            event = {"type": state["stage"], **state, "time": now}
            self._last_event_at = now
        self._last_stage = state["stage"]
        return state, event
```

- [ ] **Step 4: Run protocol runtime and existing intervention tests to verify they pass**

Run: `pytest tests/test_protocol_runtime.py tests/test_intervention_policy.py -v`

Expected: PASS with 5 passing tests.

- [ ] **Step 5: Commit**

```bash
git add src/protocol_runtime.py src/intervention_policy.py tests/test_protocol_runtime.py
git commit -m "feat: drive live intervention from protocol runtime"
```

## Task 3: Build protocol-aware review cards, structured summaries, and hypotheses

**Files:**
- Create: `src/protocol_summary.py`
- Modify: `src/protocol_runtime.py`
- Modify: `src/session_insight.py`
- Test: `tests/test_protocol_summary.py`

- [ ] **Step 1: Write the failing review-summary tests**

```python
from src.protocol_loader import load_protocol_pack
from src.protocol_runtime import ProtocolRuntime
from src.protocol_summary import build_summary_plan, build_primary_hypothesis


def make_review():
    return {
        "summary": {
            "average_stamina": 61.4,
            "largest_drop": 28,
            "recovered": False,
            "duration_min": 10.0,
            "net_change": -18,
        },
        "moments": {
            "lowest": {"minute": 6.0, "stamina": 41},
            "steepest_drop": {"minute": 5.0, "delta": 16},
        },
        "flags": [
            {
                "type": "high_tension",
                "minutes": 3,
                "message": "张力段偏高，建议检查姿势、击键力度或休息节奏。",
            }
        ],
        "series": [],
    }


def test_protocol_runtime_builds_review_cards_from_review_payload():
    runtime = ProtocolRuntime(load_protocol_pack())

    board = runtime.build_review_board(make_review())

    assert [card["id"] for card in board["cards"]] == [
        "largest_drop",
        "high_tension_segment",
        "failed_recovery",
    ]


def test_summary_plan_and_hypothesis_follow_review_cards():
    runtime = ProtocolRuntime(load_protocol_pack())
    board = runtime.build_review_board(make_review())

    summary_plan = build_summary_plan(load_protocol_pack(), make_review(), board)
    hypothesis = build_primary_hypothesis(load_protocol_pack(), board)

    assert summary_plan["blocks"]["observation"]
    assert "张力" in summary_plan["blocks"]["interpretation"]
    assert hypothesis["template_id"] == "high_tension_recovery"
```

- [ ] **Step 2: Run the summary tests to verify they fail**

Run: `pytest tests/test_protocol_summary.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.protocol_summary'`.

- [ ] **Step 3: Write the minimal review-summary implementation**

```python
# src/protocol_runtime.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .protocol_models import ProtocolPack, ReviewCardSpec, StageSpec


class ProtocolRuntime:
    ...

    def build_review_board(self, review: Dict[str, Any]) -> Dict[str, Any]:
        cards = []
        summary = review.get("summary") or {}
        flags = list(review.get("flags") or [])

        for spec in self.pack.review.cards:
            if self._matches_review_card(spec, summary=summary, flags=flags):
                cards.append(
                    {
                        "id": spec.id,
                        "title": spec.title,
                        "severity": spec.severity,
                        "body": spec.body,
                        "suggests_experiment": spec.suggests_experiment,
                    }
                )

        return {
            "protocol_id": self.pack.protocol_id,
            "cards": cards,
        }

    def _matches_review_card(self, spec: ReviewCardSpec, *, summary: Dict[str, Any], flags: List[Dict[str, Any]]) -> bool:
        condition = spec.condition
        largest_drop_gte = condition.get("largest_drop_gte")
        recovered_is = condition.get("recovered_is")
        flag_type = condition.get("flag_type")

        if largest_drop_gte is not None and float(summary.get("largest_drop", 0.0)) < float(largest_drop_gte):
            return False
        if recovered_is is not None and bool(summary.get("recovered")) is not bool(recovered_is):
            return False
        if flag_type is not None and not any(flag.get("type") == flag_type for flag in flags):
            return False
        return True
```

```python
# src/protocol_summary.py
from __future__ import annotations

from typing import Any, Dict

from .protocol_models import ProtocolPack


def build_summary_plan(pack: ProtocolPack, review: Dict[str, Any], board: Dict[str, Any]) -> Dict[str, Any]:
    summary = review.get("summary") or {}
    cards = board.get("cards") or []
    card_ids = {card["id"] for card in cards}

    observation = f"本次会话平均耐力 {summary.get('average_stamina', 0)}，最大下滑 {summary.get('largest_drop', 0)}。"
    if summary.get("recovered"):
        observation += " 后段出现恢复。"
    else:
        observation += " 后段恢复不明显。"

    if "high_tension_segment" in card_ids:
        interpretation = "张力持续偏高，可能放大了耐力下滑。"
    elif "failed_recovery" in card_ids:
        interpretation = "明显下滑后没有恢复，说明当前节奏可能不适合持续工作。"
    else:
        interpretation = "这次会话整体较稳定，但仍可持续观察下滑触发点。"

    if "high_tension_segment" in card_ids:
        next_experiment = "下次在张力升高时加入一次 30 秒肩颈重置，并比较后续恢复。"
        confidence = "medium"
    elif "failed_recovery" in card_ids:
        next_experiment = "下次在明显下滑后提前 1 次轻提醒，观察是否能缩短恢复时间。"
        confidence = "medium"
    else:
        next_experiment = "继续记录 2 次 session，确认当前稳定状态是否可复现。"
        confidence = "low"

    return {
        "protocol_id": pack.protocol_id,
        "blocks": {
            "observation": observation,
            "interpretation": interpretation,
            "next_experiment": next_experiment,
            "confidence": confidence,
        },
    }


def build_primary_hypothesis(pack: ProtocolPack, board: Dict[str, Any]) -> Dict[str, Any]:
    card_ids = {card["id"] for card in (board.get("cards") or [])}
    for template in pack.hypothesis.templates:
        if template.when_card in card_ids:
            return {
                "template_id": template.id,
                "hypothesis": template.hypothesis,
                "next_experiment": template.next_experiment,
                "confidence": template.confidence,
            }
    return {
        "template_id": "none",
        "hypothesis": "当前证据不足以提出强假设。",
        "next_experiment": "继续收集 1 到 2 次同类 session。",
        "confidence": "low",
    }
```

```python
# src/session_insight.py
from __future__ import annotations

from typing import Any, Dict

from .protocol_loader import load_protocol_pack
from .protocol_runtime import ProtocolRuntime
from .protocol_summary import build_primary_hypothesis, build_summary_plan
from .session_review import build_session_review


def session_insight_text(pkg: Dict[str, Any]) -> str:
    review = build_session_review(pkg)
    if not review.get("series"):
        return "暂无快照，无法生成洞察。"

    pack = load_protocol_pack()
    runtime = ProtocolRuntime(pack)
    board = runtime.build_review_board(review)
    summary_plan = build_summary_plan(pack, review, board)
    hypothesis = build_primary_hypothesis(pack, board)
    blocks = summary_plan["blocks"]
    return " ".join(
        [
            blocks["observation"],
            blocks["interpretation"],
            f"建议：{blocks['next_experiment']}",
            f"假设：{hypothesis['hypothesis']}",
        ]
    )
```

- [ ] **Step 4: Run summary, review, and insight tests to verify they pass**

Run: `pytest tests/test_protocol_summary.py tests/test_session_review.py -v`

Expected: PASS with 5 passing tests.

- [ ] **Step 5: Commit**

```bash
git add src/protocol_runtime.py src/protocol_summary.py src/session_insight.py tests/test_protocol_summary.py
git commit -m "feat: add protocol-driven review summary flow"
```

## Task 4: Wire protocol metadata and evidence records into FastAPI responses

**Files:**
- Create: `src/protocol_evidence.py`
- Modify: `web/app.py`
- Create: `tests/test_protocol_api.py`
- Modify: `tests/test_session_review.py`

- [ ] **Step 1: Write the failing API tests**

```python
import json

from fastapi.testclient import TestClient

import web.app as web_app


def test_state_endpoint_returns_protocol_metadata(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "latest_payload",
        {
            "protocol_config": {
                "id": "quiet_loop_v1",
                "stages": [
                    {"id": "silent_log", "title": "Silent Logging", "reason_copy": "状态开始下滑，先静默记录，不打断工作。"},
                    {"id": "light_nudge", "title": "Gentle Nudge", "reason_copy": "状态持续走低，已到最轻提醒阈值。"},
                    {"id": "escalation", "title": "Escalation", "reason_copy": "出现强风险信号，建议暂停并进入复盘。"},
                ],
            },
            "intervention": {
                "stage": "light_nudge",
                "score": 0.61,
                "should_surface": True,
                "protocol": {
                    "id": "quiet_loop_v1",
                    "rule_id": "light_nudge",
                    "surface": "visible",
                    "reason_copy": "状态持续走低，已到最轻提醒阈值。",
                },
            }
        },
        raising=False,
    )

    with TestClient(web_app.app) as client:
        resp = client.get("/api/v1/state")

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["intervention"]["protocol"]["rule_id"] == "light_nudge"
    assert body["data"]["protocol_config"]["stages"][1]["id"] == "light_nudge"


def test_review_endpoint_returns_review_board_and_summary_plan(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr("src.session_store.SESSIONS_DIR", session_dir)
    monkeypatch.setattr(web_app, "SESSIONS_DIR", session_dir, raising=False)

    pkg = {
        "session": {"id": "12345678", "durationSec": 600},
        "snapshots": [
            {"t": 0, "stamina": 78, "ten": 0.2, "fat": 0.1, "con": 0.7, "state": "focused", "act": "typing"},
            {"t": 300, "stamina": 49, "ten": 0.68, "fat": 0.4, "con": 0.5, "state": "fading", "act": "typing"},
        ],
    }
    (session_dir / "fluxchi_20260402_120000_12345678.json").write_text(json.dumps(pkg), encoding="utf-8")

    with TestClient(web_app.app) as client:
        resp = client.get("/api/v1/sessions/12345678/review")

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["review_board"]["cards"]
    assert body["data"]["summary_plan"]["blocks"]["next_experiment"]
    assert body["data"]["hypothesis"]["confidence"]
```

- [ ] **Step 2: Run the API tests to verify they fail**

Run: `pytest tests/test_protocol_api.py -v`

Expected: FAIL because the review endpoint still returns only the raw review payload.

- [ ] **Step 3: Write the minimal evidence and API integration**

```python
# src/protocol_evidence.py
from __future__ import annotations

from typing import Any, Dict


def build_protocol_event(decision: Dict[str, Any], *, timestamp: float, session_id: str | None = None) -> Dict[str, Any]:
    protocol = dict(decision.get("protocol") or {})
    return {
        "protocol_id": protocol.get("id"),
        "rule_id": protocol.get("rule_id"),
        "session_id": session_id,
        "timestamp": timestamp,
        "stage": decision.get("stage"),
        "surface_type": protocol.get("surface"),
        "decision_context": {
            "score": decision.get("score"),
            "active_for_sec": decision.get("active_for_sec"),
        },
        "evidence_refs": list(decision.get("evidence") or []),
        "proximal_outcome": None,
    }
```

```python
# web/app.py
from src.protocol_loader import load_protocol_pack
from src.protocol_runtime import ProtocolRuntime
from src.protocol_summary import build_primary_hypothesis, build_summary_plan
from src.protocol_evidence import build_protocol_event

protocol_pack = load_protocol_pack()
protocol_runtime = ProtocolRuntime(protocol_pack)


def _protocol_config_payload() -> Dict[str, Any]:
    return {
        "id": protocol_pack.protocol_id,
        "stages": [
            {
                "id": stage.id,
                "title": stage.title,
                "reason_copy": stage.reason_copy,
                "surface": stage.surface,
            }
            for stage in protocol_pack.intervention.stages
        ],
    }


def _build_review_bundle(pkg: Dict[str, Any]) -> Dict[str, Any]:
    review = build_session_review(pkg)
    board = protocol_runtime.build_review_board(review)
    summary_plan = build_summary_plan(protocol_pack, review, board)
    hypothesis = build_primary_hypothesis(protocol_pack, board)
    return {
        **review,
        "review_board": board,
        "summary_plan": summary_plan,
        "hypothesis": hypothesis,
    }


@app.get("/api/v1/sessions/{session_id}/review", tags=["Sessions"], summary="读取结构化复盘")
async def v1_sessions_review(session_id: str):
    try:
        pkg = load_session(session_id)
    except ValueError:
        return _err("bad_request", "非法 session_id")
    except FileNotFoundError:
        return _err("not_found", "会话不存在")
    return _ok(_build_review_bundle(pkg))
```

```python
# web/app.py (inside processing_loop where intervention state is attached)
state, event = intervention_tracker.update(payload, now=now_ts)
payload["intervention"] = state
payload["protocol_config"] = _protocol_config_payload()

if event:
    protocol_event = build_protocol_event(event, timestamp=now_ts)
    timeline.append({**event, "protocol_event": protocol_event})
    if web_recording is not None:
        web_recording.setdefault("protocol_events", []).append(protocol_event)
```

```python
# web/app.py (inside v1_sessions_web_stop after pkg is built)
review_bundle = _build_review_bundle(pkg)
pkg["protocol"] = {
    "review_board": review_bundle["review_board"],
    "summary_plan": review_bundle["summary_plan"],
    "hypothesis": review_bundle["hypothesis"],
    "events": list(rec.get("protocol_events") or []),
}
pkg["summary"]["text"] = session_insight_text(pkg)
```

- [ ] **Step 4: Run all protocol and session API tests to verify they pass**

Run: `pytest tests/test_protocol_api.py tests/test_session_review.py tests/test_intervention_policy.py tests/test_protocol_runtime.py tests/test_protocol_summary.py -v`

Expected: PASS with all protocol-related tests passing.

- [ ] **Step 5: Commit**

```bash
git add src/protocol_evidence.py web/app.py tests/test_protocol_api.py tests/test_session_review.py
git commit -m "feat: expose protocol loop through web api"
```

## Task 5: Make the Web UI render protocol-driven review and summary content

**Files:**
- Create: `web/static/protocol-ui.js`
- Modify: `web/static/dashboard.js`
- Modify: `web/static/dashboard.css`
- Modify: `web/static/index.html`

- [ ] **Step 1: Add the protocol UI helper module**

```javascript
// web/static/protocol-ui.js
export function stageClass(stage) {
  return ["steady", "silent_log", "light_nudge", "escalation", "recovered"].includes(stage)
    ? stage
    : "steady";
}

export function protocolTitle(intervention) {
  return intervention?.protocol?.title || "Quiet Monitoring";
}

export function protocolReason(intervention, fallbackReasons = []) {
  if (intervention?.protocol?.reason_copy) return intervention.protocol.reason_copy;
  if (Array.isArray(intervention?.evidence) && intervention.evidence.length) return intervention.evidence.join(" · ");
  return fallbackReasons.join(" · ");
}

export function renderProtocolRail(container, stages, activeStage) {
  container.innerHTML = stages
    .map((stage) => {
      const active = stage.id === activeStage ? " active" : "";
      return `<div class="rail-step${active}">
        <strong>${stage.title}</strong>
        <p>${stage.reason_copy}</p>
      </div>`;
    })
    .join("");
}

export function renderSummaryBlocks(container, summaryPlan, hypothesis) {
  const blocks = summaryPlan?.blocks || {};
  container.innerHTML = `
    <div class="summary-block">
      <span>Observation</span>
      <strong>${blocks.observation || "—"}</strong>
    </div>
    <div class="summary-block">
      <span>Interpretation</span>
      <strong>${blocks.interpretation || "—"}</strong>
    </div>
    <div class="summary-block">
      <span>Next Experiment</span>
      <strong>${blocks.next_experiment || "—"}</strong>
    </div>
    <div class="summary-block">
      <span>Confidence</span>
      <strong>${blocks.confidence || hypothesis?.confidence || "low"}</strong>
    </div>
  `;
}

export function renderReviewCards(container, reviewBoard) {
  const cards = reviewBoard?.cards || [];
  container.innerHTML = cards.length
    ? cards
        .map((card) => `<div class="protocol-card ${card.severity}">
          <strong>${card.title}</strong>
          <p>${card.body}</p>
        </div>`)
        .join("")
    : `<div class="placeholder">No protocol cards yet.</div>`;
}
```

- [ ] **Step 2: Update the dashboard to consume backend protocol fields**

```javascript
// web/static/dashboard.js
import { createVisionCapture } from "/static/vision-capture.js";
import {
  protocolReason,
  protocolTitle,
  renderProtocolRail,
  renderReviewCards,
  renderSummaryBlocks,
  stageClass,
} from "/static/protocol-ui.js";

const appState = {
  sessions: [],
  selectedSessionId: null,
  livePayload: null,
  visionCapture: null,
  protocolStages: [],
};

function renderLive(payload) {
  appState.livePayload = payload;
  if (payload?.protocol_config?.stages?.length) {
    appState.protocolStages = payload.protocol_config.stages;
  }
  const intervention = payload?.intervention || { stage: "steady", evidence: [] };
  const stage = stageClass(intervention.stage);
  const score = payload?.fusion?.stamina ?? payload?.stamina?.value;

  $("liveStage").textContent = intervention?.protocol?.rule_id || stage;
  $("liveStage").className = `stage-pill ${stage}`;
  $("heroScore").textContent = typeof score === "number" ? Math.round(score) : "--";
  $("heroState").textContent = protocolTitle(intervention);
  $("heroReason").textContent = protocolReason(intervention, payload?.decision?.reasons || []);
  $("liveStageMeta").textContent = intervention?.protocol?.surface === "visible" ? "visible intervention" : "silent by default";

  renderProtocolRail($("protocolRail"), appState.protocolStages, stage);
  renderMetricCards(payload?.stamina);
  renderSignalSummary(payload);
}

async function loadReview(sessionId) {
  const resp = await fetch(`/api/v1/sessions/${sessionId}/review`);
  const body = await resp.json();
  const review = body?.ok ? body.data : null;
  renderReviewBoard(review, appState.sessions.find((session) => session.id === sessionId));
  renderReviewCards($("reviewCards"), review?.review_board);
  renderSummaryBlocks($("summaryBlocks"), review?.summary_plan, review?.hypothesis);
}
```

- [ ] **Step 3: Add the new HTML slots and styles**

```html
<!-- web/static/index.html -->
<section class="panel">
  <div class="panel-head">
    <h3>Protocol Review</h3>
  </div>
  <div id="reviewCards" class="protocol-card-list"></div>
</section>

<section class="panel">
  <div class="panel-head">
    <h3>Post-Session Summary</h3>
  </div>
  <div id="summaryBlocks" class="summary-block-grid"></div>
</section>
```

```css
/* web/static/dashboard.css */
.protocol-card-list,
.summary-block-grid {
  display: grid;
  gap: 12px;
}

.protocol-card,
.summary-block {
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(9, 15, 22, 0.72);
  border-radius: 16px;
  padding: 14px 16px;
}

.protocol-card.warn {
  border-color: rgba(246, 194, 62, 0.35);
}

.protocol-card.danger {
  border-color: rgba(255, 108, 88, 0.4);
}

.summary-block span {
  display: block;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(212, 226, 238, 0.62);
}

.summary-block strong {
  display: block;
  margin-top: 6px;
  font-size: 15px;
  line-height: 1.5;
  color: #f3f7fa;
}
```

- [ ] **Step 4: Run protocol tests, start the demo server, and manually verify the UI**

Run: `pytest tests/test_protocol_loader.py tests/test_protocol_runtime.py tests/test_protocol_summary.py tests/test_protocol_api.py tests/test_intervention_policy.py tests/test_session_review.py -v`

Expected: PASS with all protocol-related tests passing.

Run: `python web/app.py --demo --web-port 18080`

Expected: server starts successfully and serves the dashboard on `http://localhost:18080`.

Manual verification checklist:

- live hero reads protocol title and reason from backend data
- protocol rail is rendered from stage data, not hard-coded English copy
- session review page shows protocol review cards
- summary panel shows `Observation`, `Interpretation`, `Next Experiment`, and `Confidence`
- archived session review still loads for prior saved sessions

- [ ] **Step 5: Commit**

```bash
git add web/static/protocol-ui.js web/static/dashboard.js web/static/dashboard.css web/static/index.html
git commit -m "feat: render protocol-driven quiet loop in web ui"
```

## Final Verification

- [ ] Run the full focused test suite:

```bash
pytest tests/test_protocol_loader.py tests/test_protocol_runtime.py tests/test_protocol_summary.py tests/test_protocol_api.py tests/test_intervention_policy.py tests/test_session_review.py -v
```

Expected: PASS with all tests green.

- [ ] Run a syntax pass:

```bash
python -m py_compile src/protocol_models.py src/protocol_loader.py src/protocol_runtime.py src/protocol_summary.py src/protocol_evidence.py src/intervention_policy.py src/session_insight.py web/app.py
```

Expected: no output and exit code `0`.

- [ ] Launch the demo app and verify the main flows:

```bash
python web/app.py --demo --web-port 18080
```

Expected flows:

- `/api/v1/state` returns `intervention.protocol`
- `/api/v1/timeline` returns protocol-aware intervention events
- `/api/v1/sessions/{id}/review` returns raw review plus `review_board`, `summary_plan`, and `hypothesis`
- the dashboard renders protocol review cards and summary blocks without hard-coded stage copy

- [ ] Final commit if verification changed anything:

```bash
git add config/fluxchi/protocols/quiet_loop_v1.yaml src/protocol_models.py src/protocol_loader.py src/protocol_runtime.py src/protocol_summary.py src/protocol_evidence.py src/intervention_policy.py src/session_insight.py web/app.py web/static/protocol-ui.js web/static/dashboard.js web/static/dashboard.css web/static/index.html tests/test_protocol_loader.py tests/test_protocol_runtime.py tests/test_protocol_summary.py tests/test_protocol_api.py tests/test_intervention_policy.py tests/test_session_review.py
git commit -m "feat: add protocol-driven quiet loop"
```
