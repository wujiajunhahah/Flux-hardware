from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TargetProfile:
    name: str
    base_url: str
    allow_writes: bool
    allow_capture: bool
    allow_shadow_read: bool
    default_headers: dict[str, str] = field(default_factory=dict)
    seed_variables: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> "TargetProfile":
        return cls(
            name=name,
            base_url=str(payload["base_url"]).rstrip("/"),
            allow_writes=bool(payload["allow_writes"]),
            allow_capture=bool(payload["allow_capture"]),
            allow_shadow_read=bool(payload["allow_shadow_read"]),
            default_headers=dict(payload.get("default_headers", {})),
            seed_variables=dict(payload.get("seed_variables", {})),
        )


def load_target_profiles(path: str | Path) -> dict[str, TargetProfile]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("target profiles file must contain an object")
    return {name: TargetProfile.from_dict(name, payload) for name, payload in raw.items()}


def ensure_operation_allowed(
    profile: TargetProfile,
    *,
    operation: str,
    is_mutating: bool,
    break_glass: bool,
) -> None:
    if not is_mutating:
        return
    if profile.allow_writes or break_glass:
        return
    raise PermissionError(f"operation {operation} is blocked for target profile {profile.name}")


@dataclass(frozen=True)
class TransportRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | None = None
    timeout_sec: float = 10.0


@dataclass(frozen=True)
class TransportResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


class Transport(ABC):
    @abstractmethod
    def request(self, request: TransportRequest) -> TransportResponse:
        raise NotImplementedError


def resolve_ca_bundle_path() -> str | None:
    explicit_path = os.getenv("SSL_CERT_FILE")
    if explicit_path:
        return explicit_path
    try:
        import certifi
    except Exception:
        return None
    return certifi.where()


class LiveTransport(Transport):
    def request(self, request: TransportRequest) -> TransportResponse:
        prepared = urllib.request.Request(
            request.url,
            data=request.body,
            headers=request.headers,
            method=request.method,
        )
        ssl_context = None
        if request.url.startswith("https://"):
            ca_bundle_path = resolve_ca_bundle_path()
            if ca_bundle_path:
                ssl_context = ssl.create_default_context(cafile=ca_bundle_path)
        try:
            with urllib.request.urlopen(prepared, timeout=request.timeout_sec, context=ssl_context) as response:
                return TransportResponse(
                    status_code=response.status,
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except urllib.error.HTTPError as exc:
            return TransportResponse(
                status_code=exc.code,
                headers=dict(exc.headers.items()),
                body=exc.read(),
            )


class FakeTransport(Transport):
    def __init__(self, responses: dict[tuple[str, str], TransportResponse] | None = None):
        self._responses = responses or {}
        self.requests: list[TransportRequest] = []

    def add_response(self, method: str, url: str, response: TransportResponse) -> None:
        self._responses[(method.upper(), url)] = response

    def request(self, request: TransportRequest) -> TransportResponse:
        self.requests.append(request)
        key = (request.method.upper(), request.url)
        try:
            return self._responses[key]
        except KeyError as exc:
            raise KeyError(f"no fake response registered for {request.method} {request.url}") from exc
