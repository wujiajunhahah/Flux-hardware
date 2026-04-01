from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

import pytest

from server.app.testing.transport import (
    FakeTransport,
    LiveTransport,
    TargetProfile,
    Transport,
    TransportRequest,
    TransportResponse,
    ensure_operation_allowed,
    load_target_profiles,
    resolve_ca_bundle_path,
)


def _example_targets_path() -> Path:
    return Path(__file__).resolve().parents[2] / "cases" / "targets.example.json"


def test_example_target_profiles_define_staging_and_production_defaults():
    profiles = load_target_profiles(_example_targets_path())

    assert set(profiles) >= {"staging", "production"}
    assert profiles["staging"].allow_writes is True
    assert profiles["production"].allow_writes is False
    assert profiles["production"].allow_capture is False
    assert profiles["production"].allow_shadow_read is True


def test_production_writes_require_explicit_break_glass_override():
    production = TargetProfile(
        name="production",
        base_url="https://api.focux.me",
        allow_writes=False,
        allow_capture=False,
        allow_shadow_read=True,
    )

    with pytest.raises(PermissionError):
        ensure_operation_allowed(production, operation="sessions.create", is_mutating=True, break_glass=False)

    ensure_operation_allowed(production, operation="sessions.create", is_mutating=True, break_glass=True)


def test_staging_allows_mutating_operations():
    staging = TargetProfile(
        name="staging",
        base_url="https://staging.api.focux.me",
        allow_writes=True,
        allow_capture=True,
        allow_shadow_read=False,
    )

    ensure_operation_allowed(staging, operation="sessions.create", is_mutating=True, break_glass=False)


def test_fake_and_live_transports_share_the_same_interface():
    fake = FakeTransport()
    live = LiveTransport()

    assert isinstance(fake, Transport)
    assert isinstance(live, Transport)


def test_resolve_ca_bundle_path_prefers_ssl_cert_file(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/tmp/custom-ca.pem")

    assert resolve_ca_bundle_path() == "/tmp/custom-ca.pem"


def test_live_transport_uses_certifi_bundle_for_https(monkeypatch, tmp_path):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    cert_path = tmp_path / "certifi.pem"
    cert_path.write_text("dummy", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "certifi", SimpleNamespace(where=lambda: str(cert_path)))

    create_default_context = Mock(return_value="ssl-context")
    monkeypatch.setattr("server.app.testing.transport.ssl.create_default_context", create_default_context)

    class _FakeResponse:
        status = 200
        headers = {"Content-Type": "application/json"}

        def read(self) -> bytes:
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    urlopen = Mock(return_value=_FakeResponse())
    monkeypatch.setattr("server.app.testing.transport.urllib.request.urlopen", urlopen)

    response = LiveTransport().request(TransportRequest(method="GET", url="https://api.focux.me/v1/health"))

    assert response == TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=b"{}")
    create_default_context.assert_called_once_with(cafile=str(cert_path))
    _, kwargs = urlopen.call_args
    assert kwargs["context"] == "ssl-context"
