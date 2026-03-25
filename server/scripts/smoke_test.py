from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from uuid import uuid4


BASE_URL = os.getenv("FLUX_BASE_URL", "http://127.0.0.1:8000").rstrip("/")



def resolve_url(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return f"{BASE_URL}{path_or_url}"



def request_json(
    method: str,
    path_or_url: str,
    payload: dict | None = None,
    token: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if extra_headers is not None:
        headers.update(extra_headers)

    request = urllib.request.Request(
        resolve_url(path_or_url),
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} for {method} {path_or_url}: {detail}", file=sys.stderr)
        raise



def request_bytes(
    method: str,
    path_or_url: str,
    payload: bytes | None = None,
    token: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> bytes:
    headers: dict[str, str] = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if extra_headers is not None:
        headers.update(extra_headers)

    request = urllib.request.Request(
        resolve_url(path_or_url),
        data=payload,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} for {method} {path_or_url}: {detail}", file=sys.stderr)
        raise



def assert_ok(response: dict, label: str) -> dict:
    if not response.get("ok"):
        raise RuntimeError(f"{label} failed: {response}")
    data = response.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"{label} returned unexpected envelope: {response}")
    print(f"ok {label}")
    return data



def main() -> int:
    suffix = uuid4().hex[:8]
    provider_token = f"dev:smoke-{suffix}"
    device_key = f"smoke-device-{suffix}"
    device_name = f"Smoke Device {suffix}"

    sign_up_payload = {
        "provider": "apple",
        "provider_token": provider_token,
        "device": {
            "client_device_key": device_key,
            "platform": "ios",
            "device_name": device_name,
            "app_version": "0.1.0",
            "os_version": "iOS 18.0",
        },
    }

    sign_up = assert_ok(request_json("POST", "/v1/auth/sign-up", sign_up_payload), "sign_up")
    user_id = sign_up["user_id"]
    device_id = sign_up["device_id"]
    refresh_token = sign_up["refresh_token"]

    sign_in = assert_ok(request_json("POST", "/v1/auth/sign-in", sign_up_payload), "sign_in")
    access_token = sign_in["access_token"]

    devices = assert_ok(request_json("GET", "/v1/devices", token=access_token), "list_devices")
    if not devices.get("items"):
        raise RuntimeError("list_devices returned no items")

    profile = assert_ok(request_json("GET", "/v1/profile", token=access_token), "get_profile")
    profile_state = profile["profile_state"]

    updated_profile = assert_ok(
        request_json(
            "PUT",
            "/v1/profile",
            payload={
                "base_version": profile_state["version"],
                "calibration_offset": 1.25,
                "estimated_accuracy": 92.5,
                "training_count": 3,
                "active_model_release_id": None,
                "summary": {
                    "retained_feedback_count": 3,
                    "avg_absolute_error": 4.2,
                },
            },
            token=access_token,
        ),
        "update_profile",
    )

    calibration = assert_ok(
        request_json(
            "GET",
            f"/v1/devices/{device_id}/calibration",
            token=access_token,
        ),
        "get_device_calibration",
    )
    device_calibration = calibration["device_calibration"]

    assert_ok(
        request_json(
            "PUT",
            f"/v1/devices/{device_id}/calibration",
            payload={
                "base_version": device_calibration["version"],
                "device_name": device_name,
                "sensor_profile": {"channels": 8, "sample_rate_hz": 1000},
                "calibration_offset": 1.5,
            },
            token=access_token,
        ),
        "update_device_calibration",
    )

    manifest = assert_ok(
        request_json(
            "GET",
            "/v1/models/manifest?platform=ios&channel=stable&device_class=iphone",
            token=access_token,
        ),
        "get_model_manifest",
    )
    if manifest.get("active_model_manifest") is not None:
        raise RuntimeError("expected no active model manifest in smoke environment")

    bootstrap = assert_ok(
        request_json(
            "GET",
            f"/v1/sync/bootstrap?device_id={device_id}&platform=ios&channel=stable",
            token=access_token,
        ),
        "get_sync_bootstrap",
    )
    if bootstrap["profile_state"]["profile_id"] != profile_state["profile_id"]:
        raise RuntimeError("bootstrap returned unexpected profile_state")
    if not bootstrap.get("device_calibrations"):
        raise RuntimeError("bootstrap returned no device_calibrations")
    if bootstrap.get("active_model_manifest") is not None:
        raise RuntimeError("expected no active model manifest in bootstrap smoke environment")

    session_id = f"ses_{uuid4().hex[:26]}"
    session_blob = {
        "session": {
            "id": session_id,
            "title": "Smoke Session",
            "source": "ios_ble",
        },
        "signalMeta": {
            "snapshotCount": 12,
        },
    }
    session_blob_bytes = json.dumps(session_blob).encode("utf-8")
    session_sha256 = __import__("hashlib").sha256(session_blob_bytes).hexdigest()

    create_session = assert_ok(
        request_json(
            "POST",
            "/v1/sessions",
            payload={
                "session_id": session_id,
                "device_id": device_id,
                "source": "ios_ble",
                "title": "Smoke Session",
                "started_at": "2026-03-24T10:00:00Z",
                "ended_at": "2026-03-24T10:10:00Z",
                "duration_sec": 600,
                "snapshot_count": 12,
                "schema_version": 1,
                "content_type": "application/json",
                "size_bytes": len(session_blob_bytes),
                "sha256": session_sha256,
            },
            token=access_token,
            extra_headers={"Idempotency-Key": f"smoke:{session_id}:upload_v1"},
        ),
        "create_session",
    )
    upload = create_session["upload"]
    request_bytes(
        "PUT",
        upload["upload_url"],
        payload=session_blob_bytes,
        extra_headers={"Content-Type": upload["content_type"]},
    )
    print("ok upload_session_blob")

    finalized = assert_ok(
        request_json(
            "PATCH",
            f"/v1/sessions/{session_id}",
            payload={
                "status": "ready",
                "blob": {
                    "object_key": upload["object_key"],
                    "size_bytes": len(session_blob_bytes),
                    "sha256": session_sha256,
                },
            },
            token=access_token,
        ),
        "finalize_session",
    )

    sessions = assert_ok(request_json("GET", "/v1/sessions", token=access_token), "list_sessions")
    if not sessions.get("items"):
        raise RuntimeError("list_sessions returned no items")

    assert_ok(
        request_json(
            "GET",
            f"/v1/sessions/{session_id}",
            token=access_token,
        ),
        "get_session",
    )

    download_meta = assert_ok(
        request_json(
            "GET",
            f"/v1/sessions/{session_id}/download",
            token=access_token,
        ),
        "get_session_download_url",
    )
    downloaded_blob = request_bytes("GET", download_meta["download_url"])
    if downloaded_blob != session_blob_bytes:
        raise RuntimeError("downloaded session blob does not match uploaded payload")
    print("ok download_session_blob")

    feedback_event_id = f"fbk_{uuid4().hex[:26]}"
    feedback = assert_ok(
        request_json(
            "POST",
            "/v1/feedback-events",
            payload={
                "feedback_event_id": feedback_event_id,
                "device_id": device_id,
                "session_id": session_id,
                "predicted_stamina": 61,
                "actual_stamina": 35,
                "label": "fatigued",
                "kss": 8,
                "note": "smoke test feedback",
                "created_at": "2026-03-24T12:00:00Z",
            },
            token=access_token,
            extra_headers={"Idempotency-Key": f"smoke:{session_id}:feedback_v1"},
        ),
        "create_feedback_event",
    )
    if feedback["feedback_event"]["feedback_event_id"] != feedback_event_id:
        raise RuntimeError("unexpected feedback_event_id")

    feedback_events = assert_ok(
        request_json(
            "GET",
            f"/v1/feedback-events?session_id={session_id}",
            token=access_token,
        ),
        "list_feedback_events",
    )
    if not feedback_events.get("items"):
        raise RuntimeError("list_feedback_events returned no items")

    assert_ok(
        request_json(
            "POST",
            "/v1/auth/refresh",
            payload={"refresh_token": refresh_token},
        ),
        "refresh",
    )

    assert_ok(
        request_json(
            "POST",
            "/v1/auth/sign-out",
            payload={"refresh_token": refresh_token},
        ),
        "sign_out",
    )

    print(
        json.dumps(
            {
                "ok": True,
                "base_url": BASE_URL,
                "user_id": user_id,
                "device_id": device_id,
                "profile_version": updated_profile["profile_state"]["version"],
                "session_id": session_id,
                "feedback_event_id": feedback_event_id,
                "download_url": finalized["session"]["download_url"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
