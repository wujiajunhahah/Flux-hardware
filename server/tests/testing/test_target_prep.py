from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.app.testing.target_prep import (
    build_shadow_auth_payload,
    load_target_profile,
    prepare_shadow_target,
    update_target_profile_auth,
)
from server.app.testing.transport import FakeTransport, TargetProfile, TransportResponse


def test_prepare_shadow_target_returns_sign_up_result():
    profile = TargetProfile(
        name="production",
        base_url="https://api.focux.me",
        allow_writes=False,
        allow_capture=False,
        allow_shadow_read=True,
    )
    transport = FakeTransport(
        {
            (
                "POST",
                "https://api.focux.me/v1/auth/sign-up",
            ): TransportResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body=json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "user_id": "usr_01",
                            "device_id": "dev_01",
                            "access_token": "token_01",
                            "refresh_token": "refresh_01",
                            "expires_in_sec": 3600,
                        },
                    }
                ).encode("utf-8"),
            )
        }
    )

    prepared = prepare_shadow_target(
        profile=profile,
        payload=build_shadow_auth_payload(
            provider="apple",
            provider_token="dev:shadow",
            client_device_key="shadow-device",
            platform="ios",
            device_name="Shadow Device",
            app_version="0.1.0",
            os_version="iOS 18.0",
        ),
        transport=transport,
    )

    assert prepared.auth_action == "sign_up"
    assert prepared.device_id == "dev_01"
    assert len(transport.requests) == 1


def test_prepare_shadow_target_falls_back_to_sign_in_on_identity_exists():
    profile = TargetProfile(
        name="production",
        base_url="https://api.focux.me",
        allow_writes=False,
        allow_capture=False,
        allow_shadow_read=True,
    )
    transport = FakeTransport(
        {
            (
                "POST",
                "https://api.focux.me/v1/auth/sign-up",
            ): TransportResponse(
                status_code=409,
                headers={"Content-Type": "application/json"},
                body=json.dumps(
                    {
                        "ok": False,
                        "error": {
                            "code": "identity_already_exists",
                            "message": "identity already exists",
                        },
                    }
                ).encode("utf-8"),
            ),
            (
                "POST",
                "https://api.focux.me/v1/auth/sign-in",
            ): TransportResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body=json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "user_id": "usr_02",
                            "device_id": "dev_02",
                            "access_token": "token_02",
                            "refresh_token": "refresh_02",
                            "expires_in_sec": 3600,
                        },
                    }
                ).encode("utf-8"),
            ),
        }
    )

    prepared = prepare_shadow_target(
        profile=profile,
        payload=build_shadow_auth_payload(
            provider="apple",
            provider_token="dev:shadow",
            client_device_key="shadow-device",
            platform="ios",
            device_name="Shadow Device",
            app_version="0.1.0",
            os_version="iOS 18.0",
        ),
        transport=transport,
    )

    assert prepared.auth_action == "sign_in"
    assert prepared.device_id == "dev_02"
    assert len(transport.requests) == 2


def test_update_target_profile_auth_writes_token_and_device_id(tmp_path: Path):
    targets_path = tmp_path / "targets.local.json"
    targets_path.write_text(
        json.dumps(
            {
                "production": {
                    "base_url": "https://api.focux.me",
                    "allow_writes": False,
                    "allow_capture": False,
                    "allow_shadow_read": True,
                    "default_headers": {"X-Trace": "1"},
                    "seed_variables": {"existing": "keep-me"},
                }
            }
        ),
        encoding="utf-8",
    )

    profile = load_target_profile(targets_path, "production")
    prepared = prepare_shadow_target(
        profile=profile,
        payload=build_shadow_auth_payload(
            provider="apple",
            provider_token="dev:shadow",
            client_device_key="shadow-device",
            platform="ios",
            device_name="Shadow Device",
            app_version="0.1.0",
            os_version="iOS 18.0",
        ),
        transport=FakeTransport(
            {
                (
                    "POST",
                    "https://api.focux.me/v1/auth/sign-up",
                ): TransportResponse(
                    status_code=200,
                    headers={"Content-Type": "application/json"},
                    body=json.dumps(
                        {
                            "ok": True,
                            "data": {
                                "user_id": "usr_03",
                                "device_id": "dev_03",
                                "access_token": "token_03",
                                "refresh_token": "refresh_03",
                                "expires_in_sec": 3600,
                            },
                        }
                    ).encode("utf-8"),
                )
            }
        ),
    )

    updated = update_target_profile_auth(
        targets_path=targets_path,
        target_name="production",
        prepared=prepared,
    )

    assert updated["production"]["default_headers"]["Authorization"] == "Bearer token_03"
    assert updated["production"]["default_headers"]["X-Trace"] == "1"
    assert updated["production"]["seed_variables"]["device_id"] == "dev_03"
    assert updated["production"]["seed_variables"]["existing"] == "keep-me"
