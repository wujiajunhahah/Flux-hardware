from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from .transport import TransportRequest, TransportResponse

MAX_CASE_ARTIFACT_BYTES = 1_000_000
MAX_RUN_ARTIFACT_BYTES = 10_000_000
MAX_PAYLOAD_BYTES = 131_072

_FAILURE_PAYLOAD_FILES = {
    "session_blob": "session_blob.json",
    "finalize": "finalize_payload.json",
    "feedback": "feedback_payload.json",
}


class ArtifactWriter:
    def __init__(
        self,
        *,
        artifacts_root: Path,
        run_id: str,
        max_case_artifact_bytes: int = MAX_CASE_ARTIFACT_BYTES,
        max_run_artifact_bytes: int = MAX_RUN_ARTIFACT_BYTES,
        max_payload_bytes: int = MAX_PAYLOAD_BYTES,
    ):
        self.max_case_artifact_bytes = max_case_artifact_bytes
        self.max_run_artifact_bytes = max_run_artifact_bytes
        self.max_payload_bytes = max_payload_bytes

        self.run_dir = Path(artifacts_root) / "contracts" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._run_bytes = 0
        self._case_bytes: dict[str, int] = {}

    def write_case_summary(self, *, case_id: str, payload: Any) -> Path:
        path = self.run_dir / "cases" / case_id / "summary.json"
        self._write_json(case_id=case_id, path=path, payload=payload)
        return path

    def write_run_summary(self, payload: Any) -> Path:
        path = self.run_dir / "run.json"
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def write_step_http(
        self,
        *,
        case_id: str,
        step_id: str,
        request: TransportRequest | None,
        response: TransportResponse | None,
    ) -> None:
        if request is not None:
            self._write_json(
                case_id=case_id,
                path=self._step_dir(case_id, step_id) / "request.json",
                payload={
                    "method": request.method,
                    "url": request.url,
                    "headers": request.headers,
                    "body_utf8": request.body.decode("utf-8", errors="replace") if request.body else None,
                    "timeout_sec": request.timeout_sec,
                },
            )

        if response is not None:
            self._write_json(
                case_id=case_id,
                path=self._step_dir(case_id, step_id) / "response.json",
                payload={
                    "status_code": response.status_code,
                    "headers": response.headers,
                    "body_utf8": response.body.decode("utf-8", errors="replace"),
                },
            )

    def write_failure_payloads(
        self,
        *,
        case_id: str,
        step_id: str,
        payloads: Mapping[str, Any],
    ) -> None:
        step_dir = self._step_dir(case_id, step_id)
        for key, filename in _FAILURE_PAYLOAD_FILES.items():
            if key not in payloads:
                continue
            self._write_json(case_id=case_id, path=step_dir / filename, payload=payloads[key])

    def _step_dir(self, case_id: str, step_id: str) -> Path:
        step_dir = self.run_dir / "cases" / case_id / "steps" / step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def _write_json(self, *, case_id: str, path: Path, payload: Any) -> None:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        to_store = raw
        if len(raw) > self.max_payload_bytes:
            to_store = json.dumps(
                {
                    "_policy": "hashed_truncated",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "original_bytes": len(raw),
                    "truncated_preview": raw[: self.max_payload_bytes].decode("utf-8", errors="replace"),
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ).encode("utf-8")

        projected_case = self._case_bytes.get(case_id, 0) + len(to_store)
        projected_run = self._run_bytes + len(to_store)
        if projected_case > self.max_case_artifact_bytes or projected_run > self.max_run_artifact_bytes:
            to_store = json.dumps(
                {
                    "_policy": "budget_guard",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "original_bytes": len(raw),
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ).encode("utf-8")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(to_store)

        self._case_bytes[case_id] = self._case_bytes.get(case_id, 0) + len(to_store)
        self._run_bytes += len(to_store)


def cleanup_stale_run_artifacts(artifacts_root: Path, *, max_age_seconds: int) -> list[Path]:
    contracts_dir = Path(artifacts_root) / "contracts"
    if not contracts_dir.exists():
        return []

    now = time.time()
    removed: list[Path] = []
    for child in contracts_dir.iterdir():
        if not child.is_dir():
            continue
        age_seconds = now - child.stat().st_mtime
        if age_seconds <= max_age_seconds:
            continue
        shutil.rmtree(child)
        removed.append(child)
    return removed
