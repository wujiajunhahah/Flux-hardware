from __future__ import annotations

import pytest

from server.app.testing.models import CaseStep, StepResult
from server.app.testing.transport import TransportResponse

from server.app.testing.expectations import ExpectationFailure, evaluate_step_expectations


def _step(expect: dict) -> CaseStep:
    return CaseStep.model_validate(
        {
            "step_id": "create_session",
            "operation": "sessions.create",
            "input": {},
            "expect": expect,
        }
    )


def _json_response(payload: dict, status_code: int = 200) -> TransportResponse:
    import json

    return TransportResponse(
        status_code=status_code,
        headers={"content-type": "application/json"},
        body=json.dumps(payload).encode("utf-8"),
    )


def test_expectation_evaluator_supports_status_ok_json_path_rules_and_blob_equality():
    step = _step(
        {
            "status": 200,
            "ok": True,
            "json_paths": [
                {"path": "$.data.upload.upload_url", "exists": True},
                {"path": "$.data.upload.upload_url", "equals": "https://example.com/upload"},
                {"path": "$.data.upload.token", "non_empty": True},
                {"path": "$.data.upload.upload_url", "scheme": "https"},
            ],
            "blob_fixture_ref": "downloaded_blob",
        }
    )
    result = StepResult(
        response=_json_response(
            {
                "ok": True,
                "data": {
                    "upload": {
                        "upload_url": "https://example.com/upload",
                        "token": "tok_123",
                    }
                },
            }
        ),
        parsed_json={
            "ok": True,
            "data": {
                "upload": {
                    "upload_url": "https://example.com/upload",
                    "token": "tok_123",
                }
            },
        },
    )

    evaluate_step_expectations(
        case_id="case_happy",
        step=step,
        result=result,
        fixtures={"downloaded_blob": result.response.body},
    )


def test_expectation_failure_message_includes_case_step_rule_actual_expected():
    step = _step({"status": 200})
    result = StepResult(response=_json_response({"ok": True}, status_code=500))

    with pytest.raises(ExpectationFailure) as exc:
        evaluate_step_expectations(case_id="case_500", step=step, result=result)

    message = str(exc.value)
    assert "case_id=case_500" in message
    assert "step_id=create_session" in message
    assert "rule=status" in message
    assert "expected=200" in message
    assert "actual=500" in message


def test_blob_equality_failure_reports_expected_and_actual_hashes():
    step = _step({"blob_fixture_ref": "expected_blob"})
    result = StepResult(response=_json_response({"ok": True}))

    with pytest.raises(ExpectationFailure) as exc:
        evaluate_step_expectations(
            case_id="blob_case",
            step=step,
            result=result,
            fixtures={"expected_blob": b"expected-content"},
        )

    message = str(exc.value)
    assert "case_id=blob_case" in message
    assert "step_id=create_session" in message
    assert "rule=blob_equals" in message
    assert "expected_sha256=" in message
    assert "actual_sha256=" in message
