import json

from fastapi.testclient import TestClient

import web.app as web_app
from src.session_review import build_session_review


def make_pkg(staminas, tensions=None):
    tensions = tensions or [0.2] * len(staminas)
    return {
        "session": {"id": "12345678", "durationSec": max(len(staminas) - 1, 0) * 60},
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


def test_session_review_endpoint_returns_review_payload(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr("src.session_store.SESSIONS_DIR", session_dir)
    monkeypatch.setattr(web_app, "SESSIONS_DIR", session_dir, raising=False)

    pkg = {
        "session": {"id": "12345678", "durationSec": 600},
        "snapshots": [{"t": 0, "stamina": 80, "ten": 0.2, "fat": 0.1, "con": 0.7}],
    }
    (session_dir / "fluxchi_20260402_120000_12345678.json").write_text(
        json.dumps(pkg),
        encoding="utf-8",
    )

    with TestClient(web_app.app) as client:
        resp = client.get("/api/v1/sessions/12345678/review")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["data"]["summary"]["average_stamina"] == 80.0
