from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

from server.app.testing.redaction import redact_payload


def test_redacts_authorization_headers_tokens_and_nested_secrets_but_preserves_ids():
    payload = {
        "request": {
            "method": "POST",
            "url": "https://api.example.com/v1/sessions/upload?signature=sig123&X-Amz-Signature=amzsig&X-Amz-Credential=cred123&expires=3600",
            "headers": {
                "Authorization": "Bearer access-abc",
                "x-api-key": "key-123",
            },
            "json": {
                "device_id": "dev_123",
                "session_id": "ses_123",
                "access_token": "access-abc",
                "refresh_token": "refresh-def",
                "profile": {
                    "nested": {
                        "secret": "sensitive",
                        "public": "ok",
                    }
                },
            },
        },
        "response": {
            "status_code": 401,
            "json": {
                "error": "expired token",
                "session_id": "ses_123",
                "tokens": {
                    "access_token": "rotated-access",
                    "refresh_token": "rotated-refresh",
                },
            },
        },
    }

    redacted = redact_payload(payload)

    headers = redacted["request"]["headers"]
    assert headers["Authorization"] == "<redacted>"
    assert headers["x-api-key"] == "key-123"

    request_json = redacted["request"]["json"]
    assert request_json["session_id"] == "ses_123"
    assert request_json["device_id"] == "dev_123"
    assert request_json["access_token"] == "<redacted>"
    assert request_json["refresh_token"] == "<redacted>"
    assert request_json["profile"]["nested"]["secret"] == "<redacted>"
    assert request_json["profile"]["nested"]["public"] == "ok"

    response_json = redacted["response"]["json"]
    assert response_json["session_id"] == "ses_123"
    assert response_json["tokens"]["access_token"] == "<redacted>"
    assert response_json["tokens"]["refresh_token"] == "<redacted>"

    query = parse_qs(urlsplit(redacted["request"]["url"]).query)
    assert query["signature"] == ["<redacted>"]
    assert query["X-Amz-Signature"] == ["<redacted>"]
    assert query["X-Amz-Credential"] == ["<redacted>"]
    assert query["expires"] == ["3600"]


def test_redaction_does_not_mutate_original_payload():
    payload = {
        "request": {
            "headers": {"authorization": "Bearer abc"},
            "json": {"access_token": "token"},
        }
    }

    redacted = redact_payload(payload)

    assert payload["request"]["headers"]["authorization"] == "Bearer abc"
    assert payload["request"]["json"]["access_token"] == "token"
    assert redacted["request"]["headers"]["authorization"] == "<redacted>"
    assert redacted["request"]["json"]["access_token"] == "<redacted>"
