from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse

from .models import CaseStep, StepResult


_MISSING = object()


@dataclass(slots=True)
class ExpectationFailure(AssertionError):
    case_id: str
    step_id: str
    rule: str
    expected: Any
    actual: Any
    detail: str | None = None

    def __str__(self) -> str:
        base = (
            f"case_id={self.case_id} step_id={self.step_id} "
            f"rule={self.rule} expected={self.expected!r} actual={self.actual!r}"
        )
        if self.detail:
            return f"{base} {self.detail}"
        return base


def evaluate_step_expectations(
    *,
    case_id: str,
    step: CaseStep,
    result: StepResult,
    fixtures: Mapping[str, Any] | None = None,
) -> None:
    fixtures = fixtures or {}
    expectations = step.expect

    response = result.response
    payload = result.parsed_json
    if payload is None and response is not None:
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except Exception:
            payload = None

    if expectations.status is not None:
        actual = response.status_code if response is not None else None
        _assert_equal(case_id, step.step_id, "status", expectations.status, actual)

    if expectations.ok is not None:
        actual_ok = payload.get("ok") if isinstance(payload, dict) else None
        _assert_equal(case_id, step.step_id, "ok", expectations.ok, actual_ok)

    for rule in expectations.json_paths:
        exists, actual = _resolve_json_path(payload, rule.path)

        if rule.exists is not None:
            _assert_equal(case_id, step.step_id, f"json_path_exists:{rule.path}", rule.exists, exists)

        if rule.equals is not None:
            if not exists:
                _fail(case_id, step.step_id, f"json_path_equals:{rule.path}", rule.equals, "<missing>")
            _assert_equal(case_id, step.step_id, f"json_path_equals:{rule.path}", rule.equals, actual)

        if rule.non_empty is not None:
            if not exists:
                _fail(case_id, step.step_id, f"json_path_non_empty:{rule.path}", rule.non_empty, "<missing>")
            actual_non_empty = _is_non_empty(actual)
            _assert_equal(
                case_id,
                step.step_id,
                f"json_path_non_empty:{rule.path}",
                rule.non_empty,
                actual_non_empty,
            )

        if rule.scheme is not None:
            if not exists:
                _fail(case_id, step.step_id, f"json_path_scheme:{rule.path}", rule.scheme, "<missing>")
            actual_scheme = urlparse(str(actual)).scheme
            _assert_equal(case_id, step.step_id, f"json_path_scheme:{rule.path}", rule.scheme, actual_scheme)

    if expectations.blob_fixture_ref is not None:
        ref = expectations.blob_fixture_ref
        expected_blob = _to_bytes(fixtures.get(ref, _MISSING))
        actual_blob = _to_bytes(response.body if response is not None else _MISSING)
        if expected_blob is _MISSING:
            _fail(case_id, step.step_id, "blob_equals", ref, "<fixture-missing>")
        if actual_blob is _MISSING:
            _fail(case_id, step.step_id, "blob_equals", "<response.body>", "<missing>")
        expected_sha = hashlib.sha256(expected_blob).hexdigest()
        actual_sha = hashlib.sha256(actual_blob).hexdigest()
        if expected_sha != actual_sha:
            raise ExpectationFailure(
                case_id=case_id,
                step_id=step.step_id,
                rule="blob_equals",
                expected=ref,
                actual="response.body",
                detail=f"expected_sha256={expected_sha} actual_sha256={actual_sha}",
            )


def _assert_equal(case_id: str, step_id: str, rule: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        _fail(case_id, step_id, rule, expected, actual)


def _fail(case_id: str, step_id: str, rule: str, expected: Any, actual: Any) -> None:
    raise ExpectationFailure(case_id=case_id, step_id=step_id, rule=rule, expected=expected, actual=actual)


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, bytearray, list, dict, tuple, set)):
        return len(value) > 0
    return True


def _to_bytes(value: Any) -> bytes | object:
    if value is _MISSING:
        return _MISSING
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _resolve_json_path(payload: Any, path: str) -> tuple[bool, Any]:
    if payload is None:
        return False, None
    if path == "$":
        return True, payload
    if not path.startswith("$."):
        return False, None

    current: Any = payload
    for token in _parse_path_tokens(path[2:]):
        if isinstance(token, int):
            if not isinstance(current, list) or token < 0 or token >= len(current):
                return False, None
            current = current[token]
            continue
        if not isinstance(current, dict) or token not in current:
            return False, None
        current = current[token]

    return True, current


def _parse_path_tokens(raw: str) -> list[str | int]:
    tokens: list[str | int] = []
    for segment in raw.split("."):
        if not segment:
            continue
        head = segment.split("[", 1)[0]
        if head:
            tokens.append(head)
        rest = segment[len(head) :]
        while rest:
            if not rest.startswith("["):
                break
            end = rest.find("]")
            if end < 0:
                break
            idx = rest[1:end]
            if idx.isdigit():
                tokens.append(int(idx))
            rest = rest[end + 1 :]
    return tokens
