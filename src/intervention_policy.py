"""Deterministic intervention policy for the web cockpit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


SEVERE_ALERTS = {"perclos_severe", "microsleep_detected", "nod_detected"}


@dataclass
class InterventionTracker:
    """Track low-friction intervention stages over time.

    The product rule is:
    - first decline: silent log only
    - persistent decline: light nudge
    - severe evidence: escalation
    """

    sustain_sec: float = 180.0
    nudge_sec: float = 300.0
    cooldown_sec: float = 240.0
    _decline_started_at: Optional[float] = None
    _last_stage: str = "steady"
    _last_event_at: float = 0.0

    def update(self, payload: Dict[str, Any], now: float) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        stamina_block = payload.get("stamina") or {}
        decision = payload.get("decision") or {}
        fusion = payload.get("fusion") or {}

        stamina = float(stamina_block.get("value") or 0.0)
        urgency = float(decision.get("urgency") or 0.0)
        reasons = list(decision.get("reasons") or [])
        alerts = list(fusion.get("alerts") or [])

        severe = any(alert in SEVERE_ALERTS for alert in alerts)
        declining = severe or urgency >= 0.35 or stamina <= 55.0

        if declining and self._decline_started_at is None:
            self._decline_started_at = now

        if not declining:
            previous_stage = self._last_stage
            self._decline_started_at = None
            self._last_stage = "steady"
            state = {
                "stage": "steady",
                "score": 0.0,
                "active_for_sec": 0.0,
                "evidence": [],
                "should_surface": False,
                "next_step": "keep_sensing",
            }
            if previous_stage in {"silent_log", "light_nudge", "escalation"}:
                return state, {"type": "recovered", **state, "time": now}
            return state, None

        active_for = now - (self._decline_started_at or now)
        evidence = alerts or reasons
        score = round(max(urgency, max(0.0, 1.0 - stamina / 100.0)), 3)

        if severe or urgency >= 0.85 or stamina <= 25.0:
            stage = "escalation"
            next_step = "pause_and_review"
        elif active_for >= self.nudge_sec or urgency >= 0.55:
            stage = "light_nudge"
            next_step = "nudge_once"
        else:
            stage = "silent_log"
            next_step = "keep_logging"

        state = {
            "stage": stage,
            "score": score,
            "active_for_sec": round(active_for, 1),
            "evidence": evidence,
            "should_surface": stage in {"light_nudge", "escalation"},
            "next_step": next_step,
        }

        event = None
        should_emit = stage != self._last_stage and (
            self._last_event_at == 0.0
            or stage == "escalation"
            or (now - self._last_event_at) >= self.cooldown_sec
        )
        if should_emit:
            event = {"type": stage, **state, "time": now}
            self._last_event_at = now
        self._last_stage = stage

        return state, event
