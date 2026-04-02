from src.intervention_policy import InterventionTracker


def make_payload(stamina: float, urgency: float, *, alerts=None, reasons=None):
    return {
        "stamina": {"value": stamina, "state": "fading" if stamina < 60 else "focused"},
        "decision": {
            "urgency": urgency,
            "recommendation": "keep_working" if urgency < 0.5 else "take_break",
            "reasons": reasons or ["效率在下降"],
        },
        "fusion": {"alerts": alerts or []},
    }


def test_first_detected_decline_only_creates_silent_log_event():
    tracker = InterventionTracker(sustain_sec=180, nudge_sec=300, cooldown_sec=240)

    state, event = tracker.update(make_payload(54.0, 0.42), now=100.0)

    assert state["stage"] == "silent_log"
    assert event["type"] == "silent_log"
    assert event["should_surface"] is False


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
