from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from server.app.testing.models import LoadedCase, RunContext
from server.app.testing.operation_registry import execute_operation, get_operation_handler
from server.app.testing.transport import FakeTransport, TargetProfile, TransportResponse


def _json_response(payload: dict, status_code: int = 200) -> TransportResponse:
    return TransportResponse(
        status_code=status_code,
        headers={"Content-Type": "application/json"},
        body=json.dumps(payload).encode("utf-8"),
    )


def _make_context(tmp_path: Path, transport: FakeTransport) -> RunContext:
    return RunContext(
        run_id="run_test",
        target_profile=TargetProfile(
            name="staging",
            base_url="https://staging.api.focux.me",
            allow_writes=True,
            allow_capture=True,
            allow_shadow_read=False,
        ),
        transport=transport,
        artifacts_dir=tmp_path / "artifacts",
    )


def _make_case() -> LoadedCase:
    return LoadedCase.model_validate(
        {
            "case_id": "full_chain_happy_path_v1",
            "title": "Full chain happy path",
            "kind": "full_chain",
            "source": "manual",
            "fixtures": {
                "provider": "apple",
                "provider_token": "dev:happy01",
                "device": {
                    "client_device_key": "case-device-happy01",
                    "platform": "ios",
                    "device_name": "Contract Case Device happy01",
                    "app_version": "0.1.0",
                    "os_version": "iOS 18.0",
                },
                "session_blob": {
                    "session": {
                        "id": "ses_fixture_minimal",
                        "title": "Minimal Session Blob",
                        "source": "ios_ble",
                    },
                    "signalMeta": {
                        "snapshotCount": 12,
                    },
                },
                "feedback": {
                    "feedback_event_id": "fbk_case_01",
                    "predicted_stamina": 61,
                    "actual_stamina": 35,
                    "label": "fatigued",
                    "kss": 8,
                    "note": "contract case feedback",
                    "created_at": "2026-03-24T10:12:00Z",
                },
            },
            "steps": [
                {
                    "step_id": "sign_up",
                    "operation": "auth.sign_up",
                    "extract": {
                        "access_token": "$.data.access_token",
                        "refresh_token": "$.data.refresh_token",
                        "device_id": "$.data.device_id",
                    },
                },
                {
                    "step_id": "sign_in",
                    "operation": "auth.sign_in",
                    "extract": {
                        "access_token": "$.data.access_token",
                    },
                },
                {
                    "step_id": "refresh",
                    "operation": "auth.refresh",
                },
                {
                    "step_id": "sign_out",
                    "operation": "auth.sign_out",
                },
                {
                    "step_id": "bootstrap",
                    "operation": "sync.bootstrap",
                    "input": {
                        "platform": "ios",
                        "channel": "stable",
                    },
                },
                {
                    "step_id": "create_session",
                    "operation": "sessions.create",
                    "input": {
                        "session_id": "ses_contract_happy01",
                        "source": "ios_ble",
                        "title": "Contract Session happy01",
                        "started_at": "2026-03-24T10:00:00Z",
                        "ended_at": "2026-03-24T10:10:00Z",
                        "duration_sec": 600,
                        "snapshot_count": 12,
                        "schema_version": 1,
                        "content_type": "application/json",
                    },
                    "extract": {
                        "session_id": "$.data.session.session_id",
                        "upload_url": "$.data.upload.upload_url",
                        "object_key": "$.data.upload.object_key",
                    },
                },
                {
                    "step_id": "upload_blob",
                    "operation": "sessions.upload_blob",
                },
                {
                    "step_id": "finalize_session",
                    "operation": "sessions.finalize",
                    "input": {
                        "status": "ready",
                    },
                },
                {
                    "step_id": "create_feedback",
                    "operation": "feedback.create",
                },
                {
                    "step_id": "list_feedback",
                    "operation": "feedback.list",
                },
            ],
        }
    )


def test_auth_handlers_store_tokens_and_device_id(tmp_path: Path):
    transport = FakeTransport()
    case = _make_case()
    context = _make_context(tmp_path, transport)

    transport.add_response(
        "POST",
        "https://staging.api.focux.me/v1/auth/sign-up",
        _json_response(
            {
                "ok": True,
                "data": {
                    "user_id": "usr_01",
                    "device_id": "dev_01",
                    "access_token": "access_01",
                    "refresh_token": "refresh_01",
                    "expires_in_sec": 3600,
                },
            }
        ),
    )
    transport.add_response(
        "POST",
        "https://staging.api.focux.me/v1/auth/sign-in",
        _json_response(
            {
                "ok": True,
                "data": {
                    "user_id": "usr_01",
                    "device_id": "dev_01",
                    "access_token": "access_02",
                    "refresh_token": "refresh_01",
                    "expires_in_sec": 3600,
                },
            }
        ),
    )
    transport.add_response(
        "POST",
        "https://staging.api.focux.me/v1/auth/refresh",
        _json_response(
            {
                "ok": True,
                "data": {
                    "access_token": "access_03",
                    "expires_in_sec": 3600,
                },
            }
        ),
    )
    transport.add_response(
        "POST",
        "https://staging.api.focux.me/v1/auth/sign-out",
        _json_response({"ok": True, "data": {"revoked": True}}),
    )

    for step in case.steps[:4]:
        execute_operation(context, case, step)

    assert context.variables["device_id"] == "dev_01"
    assert context.variables["refresh_token"] == "refresh_01"
    assert context.variables["access_token"] == "access_02"
    assert get_operation_handler("auth.refresh") is not None
    assert get_operation_handler("auth.sign_out") is not None


def test_sync_bootstrap_builds_expected_query(tmp_path: Path):
    transport = FakeTransport()
    case = _make_case()
    context = _make_context(tmp_path, transport)
    context.variables.update(
        {
            "access_token": "access_02",
            "device_id": "dev_01",
        }
    )
    transport.add_response(
        "GET",
        "https://staging.api.focux.me/v1/sync/bootstrap?device_id=dev_01&platform=ios&channel=stable",
        _json_response(
            {
                "ok": True,
                "data": {
                    "server_time": "2026-03-24T10:00:00Z",
                    "profile_state": None,
                    "device_calibrations": [],
                    "active_model_manifest": None,
                },
            }
        ),
    )

    execute_operation(context, case, case.steps[4])

    request = transport.requests[-1]
    parsed = urlparse(request.url)
    assert parsed.path == "/v1/sync/bootstrap"
    assert parse_qs(parsed.query) == {
        "device_id": ["dev_01"],
        "platform": ["ios"],
        "channel": ["stable"],
    }
    assert request.headers["Authorization"] == "Bearer access_02"


def test_session_create_upload_and_finalize_flow_uses_extracted_values(tmp_path: Path):
    transport = FakeTransport()
    case = _make_case()
    context = _make_context(tmp_path, transport)
    context.variables.update(
        {
            "access_token": "access_02",
            "device_id": "dev_01",
        }
    )

    transport.add_response(
        "POST",
        "https://staging.api.focux.me/v1/sessions",
        _json_response(
            {
                "ok": True,
                "data": {
                    "session": {
                        "session_id": "ses_contract_happy01",
                        "status": "pending_upload",
                    },
                    "upload": {
                        "object_key": "sessions/dev_01/ses_contract_happy01.json",
                        "content_type": "application/json",
                        "upload_url": "https://uploads.focux.me/put/ses_contract_happy01",
                    },
                },
            }
        ),
    )
    transport.add_response(
        "PUT",
        "https://uploads.focux.me/put/ses_contract_happy01",
        TransportResponse(status_code=200, headers={}, body=b""),
    )
    transport.add_response(
        "PATCH",
        "https://staging.api.focux.me/v1/sessions/ses_contract_happy01",
        _json_response(
            {
                "ok": True,
                "data": {
                    "session": {
                        "session_id": "ses_contract_happy01",
                        "status": "ready",
                        "download_url": "https://api.focux.me/v1/sessions/ses_contract_happy01/download",
                    }
                },
            }
        ),
    )

    execute_operation(context, case, case.steps[5])
    execute_operation(context, case, case.steps[6])
    execute_operation(context, case, case.steps[7])

    assert context.variables["session_id"] == "ses_contract_happy01"
    assert context.variables["upload_url"] == "https://uploads.focux.me/put/ses_contract_happy01"
    assert context.variables["object_key"] == "sessions/dev_01/ses_contract_happy01.json"

    upload_request = transport.requests[-2]
    assert upload_request.method == "PUT"
    assert json.loads(upload_request.body.decode("utf-8"))["session"]["title"] == "Minimal Session Blob"

    finalize_request = transport.requests[-1]
    finalize_payload = json.loads(finalize_request.body.decode("utf-8"))
    assert finalize_payload["blob"]["object_key"] == "sessions/dev_01/ses_contract_happy01.json"
    assert finalize_payload["blob"]["size_bytes"] == len(upload_request.body)
    assert finalize_payload["blob"]["sha256"]


def test_session_get_and_download_follow_metadata_then_blob_request(tmp_path: Path):
    transport = FakeTransport()
    case = _make_case()
    context = _make_context(tmp_path, transport)
    context.variables.update(
        {
            "access_token": "access_02",
            "session_id": "ses_contract_happy01",
        }
    )

    transport.add_response(
        "GET",
        "https://staging.api.focux.me/v1/sessions/ses_contract_happy01",
        _json_response(
            {
                "ok": True,
                "data": {
                    "session": {
                        "session_id": "ses_contract_happy01",
                        "status": "ready",
                    }
                },
            }
        ),
    )
    transport.add_response(
        "GET",
        "https://staging.api.focux.me/v1/sessions/ses_contract_happy01/download",
        _json_response(
            {
                "ok": True,
                "data": {
                    "download_url": "https://downloads.focux.me/get/ses_contract_happy01",
                },
            }
        ),
    )
    blob_bytes = json.dumps(case.fixtures["session_blob"]).encode("utf-8")
    transport.add_response(
        "GET",
        "https://downloads.focux.me/get/ses_contract_happy01",
        TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=blob_bytes),
    )

    get_step = case.steps[5].model_copy(update={"step_id": "get_session", "operation": "sessions.get", "extract": []})
    download_step = case.steps[5].model_copy(
        update={
            "step_id": "download_blob",
            "operation": "sessions.download",
            "expect": {
                "status": 200,
                "blob_fixture_ref": "downloaded_blob",
            },
            "extract": [],
        }
    )

    get_result = execute_operation(context, case, get_step)
    download_result = execute_operation(context, case, download_step)

    assert get_result.parsed_json["data"]["session"]["status"] == "ready"
    assert download_result.response.body == blob_bytes
    assert transport.requests[-1].url == "https://downloads.focux.me/get/ses_contract_happy01"


def test_feedback_create_and_list_work(tmp_path: Path):
    transport = FakeTransport()
    case = _make_case()
    context = _make_context(tmp_path, transport)
    context.variables.update(
        {
            "access_token": "access_02",
            "device_id": "dev_01",
            "session_id": "ses_contract_happy01",
        }
    )

    transport.add_response(
        "POST",
        "https://staging.api.focux.me/v1/feedback-events",
        _json_response(
            {
                "ok": True,
                "data": {
                    "feedback_event": {
                        "feedback_event_id": "fbk_case_01",
                    }
                },
            }
        ),
    )
    transport.add_response(
        "GET",
        "https://staging.api.focux.me/v1/feedback-events?session_id=ses_contract_happy01",
        _json_response(
            {
                "ok": True,
                "data": {
                    "items": [{"feedback_event_id": "fbk_case_01"}],
                    "next_cursor": None,
                },
            }
        ),
    )

    execute_operation(context, case, case.steps[8])
    execute_operation(context, case, case.steps[9])

    create_request = transport.requests[-2]
    assert create_request.headers["Authorization"] == "Bearer access_02"
    assert "Idempotency-Key" in create_request.headers
    list_request = transport.requests[-1]
    assert list_request.url.endswith("?session_id=ses_contract_happy01")
