from __future__ import annotations

import json
from hashlib import sha256
from typing import Any
from urllib.parse import urlencode

from ..models import CaseStep, LoadedCase, RunContext, StepResult
from ..transport import TransportRequest, TransportResponse


def build_url(context: RunContext, path: str, query: dict[str, Any] | None = None) -> str:
    url = f"{context.target_profile.base_url}{path}"
    if query:
        url = f"{url}?{urlencode(query)}"
    return url


def auth_headers(context: RunContext, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    headers = dict(context.target_profile.default_headers)
    access_token = context.variables.get("access_token")
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    if extra_headers:
        headers.update(extra_headers)
    return headers


def execute_json_request(
    context: RunContext,
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> StepResult:
    body = None
    request_headers = headers or {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers = {"Content-Type": "application/json", **request_headers}
    request = TransportRequest(method=method, url=url, headers=request_headers, body=body)
    response = context.transport.request(request)
    return normalize_step_result(request, response)


def execute_bytes_request(
    context: RunContext,
    *,
    method: str,
    url: str,
    payload: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> StepResult:
    request = TransportRequest(method=method, url=url, headers=headers or {}, body=payload)
    response = context.transport.request(request)
    return normalize_step_result(request, response)


def normalize_step_result(request: TransportRequest, response: TransportResponse) -> StepResult:
    parsed_json: Any | None = None
    content_type = response.headers.get("Content-Type", "").lower()
    if response.body and ("application/json" in content_type or response.body[:1] in {b"{", b"["}):
        parsed_json = json.loads(response.body.decode("utf-8"))
    return StepResult(request=request, response=response, parsed_json=parsed_json)


def apply_extracts(result: StepResult, step: CaseStep, context: RunContext) -> StepResult:
    if result.parsed_json is None:
        return result
    extracted = {
        rule.name: json_path_get(result.parsed_json, rule.path)
        for rule in step.extract
    }
    if extracted:
        context.variables.update(extracted)
        result.extracted.update(extracted)
    return result


def json_path_get(payload: Any, path: str) -> Any:
    if not path.startswith("$."):
        raise ValueError(f"unsupported json path: {path}")
    current = payload
    for segment in path[2:].split("."):
        if not isinstance(current, dict) or segment not in current:
            raise KeyError(f"json path not found: {path}")
        current = current[segment]
    return current


def encode_session_blob(case: LoadedCase) -> tuple[bytes, str]:
    session_blob = case.fixtures["session_blob"]
    body = json.dumps(session_blob).encode("utf-8")
    return body, sha256(body).hexdigest()
