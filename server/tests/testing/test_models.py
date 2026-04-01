from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.app.testing.models import LoadedCase


def _valid_case_payload() -> dict:
    return {
        "case_id": "session_upload_requires_https_url_v1",
        "title": "Session upload returns HTTPS URL",
        "kind": "session_upload",
        "source": "manual",
        "enabled": True,
        "tags": ["core", "platform"],
        "target_profile": "staging",
        "fixtures": {
            "provider": "apple",
        },
        "steps": [
            {
                "step_id": "create_session",
                "operation": "sessions.create",
                "input": {},
                "expect": {
                    "ok": True,
                    "status": 200,
                    "json_paths": [
                        {
                            "path": "$.data.upload.upload_url",
                            "scheme": "https",
                        }
                    ],
                },
                "extract": {
                    "session_id": "$.data.session.session_id",
                },
            }
        ],
        "artifact_policy": {
            "on_fail": ["http", "step_trace"],
            "on_pass": ["summary"],
        },
    }


def test_missing_case_id_fails_validation():
    payload = _valid_case_payload()
    payload.pop("case_id")

    with pytest.raises(ValidationError):
        LoadedCase.model_validate(payload)


def test_unknown_operation_fails_validation():
    payload = _valid_case_payload()
    payload["steps"][0]["operation"] = "sessions.destroy_everything"

    with pytest.raises(ValidationError):
        LoadedCase.model_validate(payload)


def test_invalid_artifact_policy_fails_validation():
    payload = _valid_case_payload()
    payload["artifact_policy"]["on_fail"] = ["http", "database_dump"]

    with pytest.raises(ValidationError):
        LoadedCase.model_validate(payload)


def test_valid_case_with_https_expectation_loads():
    case = LoadedCase.model_validate(_valid_case_payload())

    assert case.case_id == "session_upload_requires_https_url_v1"
    assert case.steps[0].operation == "sessions.create"
    assert case.steps[0].expect.json_paths[0].scheme == "https"
