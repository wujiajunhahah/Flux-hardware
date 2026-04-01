from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.scripts.capture_contract_failure import capture_failure_bundle


def _sample_bundle() -> dict:
    return {
        "case_id": "captured_auth_refresh_20260401",
        "title": "Captured auth refresh failure",
        "kind": "auth_refresh",
        "target_profile": "staging",
        "tags": ["captured", "incident"],
        "steps": [
            {
                "step_id": "refresh",
                "operation": "auth.refresh",
                "input": {
                    "refresh_token": "refresh-secret",
                    "session_id": "ses_123",
                },
                "expect": {
                    "status": 401,
                    "json_paths": [
                        {"path": "$.error.code", "equals": "token_expired"},
                    ],
                },
                "http": {
                    "request": {
                        "url": "https://api.example.com/v1/auth/refresh?signature=sig123&expires=99",
                        "headers": {"Authorization": "Bearer access-secret"},
                    },
                    "response": {
                        "status_code": 401,
                        "json": {
                            "refresh_token": "refresh-secret",
                            "session_id": "ses_123",
                        },
                    },
                },
            }
        ],
        "seed": {"session_id": "ses_123"},
    }


def test_capture_failure_bundle_writes_redacted_case_document(tmp_path: Path):
    output = tmp_path / "server" / "cases" / "captured"
    bundle = _sample_bundle()

    case_path = capture_failure_bundle(bundle, output_dir=output)

    assert case_path.exists()
    assert case_path.parent == output

    captured = json.loads(case_path.read_text(encoding="utf-8"))
    assert captured["source"] == "captured"
    assert captured["case_id"] == "captured_auth_refresh_20260401"
    assert captured["enabled"] is False
    assert "captured" in captured["tags"]

    step = captured["steps"][0]
    assert step["input"]["session_id"] == "ses_123"
    assert step["input"]["refresh_token"] == "<redacted>"
    assert set(step) == {"step_id", "operation", "input", "expect", "extract"}

    evidence = captured["fixtures"]["failure_bundle"]["steps"][0]["http"]
    assert evidence["request"]["headers"]["Authorization"] == "<redacted>"
    assert "signature=<redacted>" in evidence["request"]["url"]
    assert evidence["response"]["json"]["refresh_token"] == "<redacted>"


def test_capture_failure_bundle_rejects_malformed_input(tmp_path: Path):
    output = tmp_path / "server" / "cases" / "captured"
    malformed = {
        "case_id": "broken_case",
        "title": "Broken",
        "kind": "auth_refresh",
        "steps": [],
    }

    with pytest.raises(ValueError, match="malformed"):
        capture_failure_bundle(malformed, output_dir=output)
