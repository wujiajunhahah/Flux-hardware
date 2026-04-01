from __future__ import annotations

from copy import deepcopy
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit


_SENSITIVE_HEADERS = {"authorization"}
_SENSITIVE_KEYS = {
    "access_token",
    "refresh_token",
    "secret",
    "signature",
    "token",
}
_SENSITIVE_QUERY_KEYS = {
    "signature",
    "x-amz-signature",
    "x-amz-credential",
    "x-amz-security-token",
    "token",
}


def redact_payload(payload: Any) -> Any:
    return _redact_value(deepcopy(payload), parent_key=None)


def _redact_value(value: Any, *, parent_key: str | None) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = key.lower()
            if parent_key == "headers" and lowered in _SENSITIVE_HEADERS:
                redacted[key] = "<redacted>"
                continue
            if lowered == "url" and isinstance(item, str):
                redacted[key] = _redact_url(item)
                continue
            if lowered in _SENSITIVE_KEYS:
                redacted[key] = "<redacted>"
                continue
            redacted[key] = _redact_value(item, parent_key=key)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item, parent_key=parent_key) for item in value]
    return value


def _redact_url(url: str) -> str:
    split = urlsplit(url)
    query_parts: list[str] = []
    for key, value in parse_qsl(split.query, keep_blank_values=True):
        if key.lower() in _SENSITIVE_QUERY_KEYS:
            query_parts.append(f"{key}=<redacted>")
        else:
            query_parts.append(f"{key}={value}")
    return urlunsplit((split.scheme, split.netloc, split.path, "&".join(query_parts), split.fragment))
