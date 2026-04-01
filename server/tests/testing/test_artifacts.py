from __future__ import annotations

import json
import os
import time
from pathlib import Path

from server.app.testing.artifacts import ArtifactWriter, cleanup_stale_run_artifacts
from server.app.testing.events import RunEventWriter
from server.app.testing.transport import TransportRequest, TransportResponse


def test_run_events_jsonl_written_with_schema_version_and_monotonic_seq(tmp_path: Path):
    writer = RunEventWriter(artifacts_root=tmp_path / "artifacts", run_id="run_001")

    first = writer.write(type="run_started", status="ok")
    second = writer.write(type="step_finished", status="ok", case_id="case_a", step_id="step_1")

    assert first.schema_version == 1
    assert first.seq == 1
    assert second.seq == 2

    events_path = tmp_path / "artifacts" / "contracts" / "run_001" / "events.jsonl"
    assert events_path.exists()

    lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert [line["seq"] for line in lines] == [1, 2]
    assert all(line["schema_version"] == 1 for line in lines)


def test_artifact_writer_persists_step_http_and_failure_domain_payloads(tmp_path: Path):
    writer = ArtifactWriter(artifacts_root=tmp_path / "artifacts", run_id="run_100")

    request = TransportRequest(
        method="POST",
        url="https://example.com/v1/sessions",
        headers={"x-test": "1"},
        body=b'{"foo":"bar"}',
        timeout_sec=5,
    )
    response = TransportResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        body=b'{"ok":true}',
    )

    writer.write_step_http(case_id="case_1", step_id="step_create", request=request, response=response)
    writer.write_failure_payloads(
        case_id="case_1",
        step_id="step_create",
        payloads={
            "session_blob": {"session": {"session_id": "ses_1"}},
            "finalize": {"session_id": "ses_1", "status": "complete"},
            "feedback": {"feedback_event_id": "fbk_1", "helpful": True},
        },
    )

    step_dir = tmp_path / "artifacts" / "contracts" / "run_100" / "cases" / "case_1" / "steps" / "step_create"
    assert (step_dir / "request.json").exists()
    assert (step_dir / "response.json").exists()
    assert (step_dir / "session_blob.json").exists()
    assert (step_dir / "finalize_payload.json").exists()
    assert (step_dir / "feedback_payload.json").exists()


def test_large_payload_is_rewritten_using_hash_and_truncation_policy(tmp_path: Path):
    writer = ArtifactWriter(
        artifacts_root=tmp_path / "artifacts",
        run_id="run_200",
        max_case_artifact_bytes=4096,
        max_run_artifact_bytes=8192,
        max_payload_bytes=96,
    )

    large_payload = {"blob": "x" * 5000}
    writer.write_failure_payloads(
        case_id="case_big",
        step_id="step_big",
        payloads={"session_blob": large_payload},
    )

    payload_path = (
        tmp_path
        / "artifacts"
        / "contracts"
        / "run_200"
        / "cases"
        / "case_big"
        / "steps"
        / "step_big"
        / "session_blob.json"
    )
    stored = json.loads(payload_path.read_text(encoding="utf-8"))

    assert stored["_policy"] == "hashed_truncated"
    assert stored["original_bytes"] > writer.max_payload_bytes
    assert len(stored["sha256"]) == 64
    assert isinstance(stored["truncated_preview"], str)


def test_cleanup_policy_removes_stale_run_directories(tmp_path: Path):
    contracts_dir = tmp_path / "artifacts" / "contracts"
    stale = contracts_dir / "old_run"
    fresh = contracts_dir / "new_run"
    stale.mkdir(parents=True)
    fresh.mkdir(parents=True)

    stale_age_seconds = 8 * 24 * 3600
    stale_time = time.time() - stale_age_seconds
    os.utime(stale, (stale_time, stale_time))

    removed = cleanup_stale_run_artifacts(tmp_path / "artifacts", max_age_seconds=7 * 24 * 3600)

    assert stale in removed
    assert not stale.exists()
    assert fresh.exists()
