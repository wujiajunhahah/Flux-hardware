from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .transport import LiveTransport, TargetProfile, Transport, TransportRequest, load_target_profiles


@dataclass(frozen=True)
class PreparedShadowTarget:
    auth_action: str
    user_id: str
    device_id: str
    access_token: str
    refresh_token: str
    expires_in_sec: int


def build_shadow_auth_payload(
    *,
    provider: str,
    provider_token: str,
    client_device_key: str,
    platform: str,
    device_name: str,
    app_version: str,
    os_version: str,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "provider_token": provider_token,
        "device": {
            "client_device_key": client_device_key,
            "platform": platform,
            "device_name": device_name,
            "app_version": app_version,
            "os_version": os_version,
        },
    }


def prepare_shadow_target(
    *,
    profile: TargetProfile,
    payload: dict[str, Any],
    transport: Transport | None = None,
) -> PreparedShadowTarget:
    active_transport = transport or LiveTransport()
    sign_up_url = f"{profile.base_url}/v1/auth/sign-up"
    sign_in_url = f"{profile.base_url}/v1/auth/sign-in"

    sign_up_response = _request_envelope(
        active_transport,
        method="POST",
        url=sign_up_url,
        payload=payload,
    )
    if sign_up_response["status_code"] < 400 and sign_up_response["body"].get("ok") is True:
        return _prepared_shadow_target("sign_up", sign_up_response["body"])

    error = sign_up_response["body"].get("error", {})
    if sign_up_response["status_code"] == 409 and error.get("code") == "identity_already_exists":
        sign_in_response = _request_envelope(
            active_transport,
            method="POST",
            url=sign_in_url,
            payload=payload,
        )
        if sign_in_response["status_code"] < 400 and sign_in_response["body"].get("ok") is True:
            return _prepared_shadow_target("sign_in", sign_in_response["body"])
        raise RuntimeError(_build_error_message(sign_in_response, method="POST", url=sign_in_url))

    raise RuntimeError(_build_error_message(sign_up_response, method="POST", url=sign_up_url))


def update_target_profile_auth(
    *,
    targets_path: str | Path,
    target_name: str,
    prepared: PreparedShadowTarget,
) -> dict[str, Any]:
    path = Path(targets_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if target_name not in raw:
        raise KeyError(f"target profile not found: {target_name}")

    profile = dict(raw[target_name])
    headers = dict(profile.get("default_headers", {}))
    headers["Authorization"] = f"Bearer {prepared.access_token}"

    seed_variables = dict(profile.get("seed_variables", {}))
    seed_variables["device_id"] = prepared.device_id

    profile["default_headers"] = headers
    profile["seed_variables"] = seed_variables
    raw[target_name] = profile
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return raw


def load_target_profile(targets_path: str | Path, target_name: str) -> TargetProfile:
    profiles = load_target_profiles(targets_path)
    try:
        return profiles[target_name]
    except KeyError as exc:
        raise KeyError(f"target profile not found: {target_name}") from exc


def _prepared_shadow_target(auth_action: str, envelope: dict[str, Any]) -> PreparedShadowTarget:
    data = envelope.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"authentication response missing data envelope: {envelope}")
    try:
        return PreparedShadowTarget(
            auth_action=auth_action,
            user_id=str(data["user_id"]),
            device_id=str(data["device_id"]),
            access_token=str(data["access_token"]),
            refresh_token=str(data["refresh_token"]),
            expires_in_sec=int(data["expires_in_sec"]),
        )
    except KeyError as exc:
        raise RuntimeError(f"authentication response missing field: {exc.args[0]}") from exc


def _request_envelope(
    transport: Transport,
    *,
    method: str,
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    request = TransportRequest(
        method=method,
        url=url,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        body=json.dumps(payload).encode("utf-8"),
    )
    response = transport.request(request)
    try:
        body = json.loads(response.body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method} {url} returned non-JSON response") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"{method} {url} returned unexpected JSON envelope")
    return {"status_code": response.status_code, "body": body}


def _build_error_message(response: dict[str, Any], *, method: str, url: str) -> str:
    body = response["body"]
    error = body.get("error", {})
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        if code and message:
            return f"{method} {url} failed: {code} ({message})"
    return f"{method} {url} failed: {body}"
