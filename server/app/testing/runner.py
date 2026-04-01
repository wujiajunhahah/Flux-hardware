from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .artifacts import ArtifactWriter, cleanup_stale_run_artifacts
from .case_loader import load_cases
from .events import RunEventWriter
from .expectations import ExpectationFailure, evaluate_step_expectations
from .models import LoadedCase, RunContext
from .operation_registry import execute_operation, is_mutating_operation
from .transport import LiveTransport, ensure_operation_allowed, load_target_profiles


def run_cases(
    *,
    contracts_dir: str | Path,
    cases_root: str | Path,
    targets_path: str | Path,
    target_name: str,
    transport=None,
    case_id: str | None = None,
    tag: str | None = None,
    break_glass: bool = False,
    artifacts_root: str | Path = "artifacts",
) -> dict:
    cases = load_cases(contracts_dir, cases_root=cases_root, case_id=case_id, tag=tag)
    profiles = load_target_profiles(targets_path)
    target_profile = profiles[target_name]
    run_id = f"run_{uuid4().hex[:12]}"
    event_writer = RunEventWriter(artifacts_root=Path(artifacts_root), run_id=run_id)
    artifact_writer = ArtifactWriter(artifacts_root=Path(artifacts_root), run_id=run_id)
    cleanup_stale_run_artifacts(Path(artifacts_root), max_age_seconds=7 * 24 * 3600)

    selected_case_ids = [case.case_id for case in cases]
    passed_case_ids: list[str] = []
    failed_case_ids: list[str] = []

    event_writer.write(
        type="run_started",
        status="ok",
        data={"target": target_name, "selected_case_ids": selected_case_ids},
    )

    active_transport = transport or LiveTransport()

    for case in cases:
        case_failed = False
        event_writer.write(type="case_started", status="ok", case_id=case.case_id)
        context = RunContext(
            run_id=run_id,
            target_profile=target_profile,
            transport=active_transport,
            variables=dict(target_profile.seed_variables),
            artifacts_dir=event_writer.run_dir,
            event_sink=event_writer,
        )

        for step in case.steps:
            event_writer.write(type="step_started", status="ok", case_id=case.case_id, step_id=step.step_id)
            result = None
            try:
                ensure_operation_allowed(
                    target_profile,
                    operation=step.operation,
                    is_mutating=is_mutating_operation(step.operation),
                    break_glass=break_glass,
                )
                result = execute_operation(context, case, step)
                evaluate_step_expectations(
                    case_id=case.case_id,
                    step=step,
                    result=result,
                    fixtures=_expectation_fixtures(case),
                )
                event_writer.write(type="step_passed", status="ok", case_id=case.case_id, step_id=step.step_id)
            except Exception as exc:
                case_failed = True
                artifact_refs = _write_failure_artifacts(
                    artifact_writer=artifact_writer,
                    case=case,
                    step=step,
                    result=result,
                )
                event_writer.write(
                    type="step_failed",
                    status="error",
                    case_id=case.case_id,
                    step_id=step.step_id,
                    message=str(exc),
                    data={"artifact_refs": artifact_refs},
                )
                failed_case_ids.append(case.case_id)
                break

        if not case_failed:
            passed_case_ids.append(case.case_id)
            summary_path = artifact_writer.write_case_summary(
                case_id=case.case_id,
                payload={
                    "case_id": case.case_id,
                    "status": "passed",
                    "steps": [step.step_id for step in case.steps],
                },
            )
            event_writer.write(
                type="artifact_written",
                status="ok",
                case_id=case.case_id,
                data={"path": str(summary_path)},
            )
            event_writer.write(type="case_finished", status="ok", case_id=case.case_id)
        else:
            summary_path = artifact_writer.write_case_summary(
                case_id=case.case_id,
                payload={
                    "case_id": case.case_id,
                    "status": "failed",
                    "steps": [step.step_id for step in case.steps],
                },
            )
            event_writer.write(
                type="artifact_written",
                status="ok",
                case_id=case.case_id,
                data={"path": str(summary_path)},
            )
            event_writer.write(type="case_finished", status="error", case_id=case.case_id)

    run_summary = {
        "run_id": run_id,
        "target": target_name,
        "selected_case_ids": selected_case_ids,
        "passed_case_ids": passed_case_ids,
        "failed_case_ids": failed_case_ids,
    }
    run_summary_path = artifact_writer.write_run_summary(run_summary)
    event_writer.write(type="artifact_written", status="ok", data={"path": str(run_summary_path)})
    event_writer.write(
        type="run_finished",
        status="ok" if not failed_case_ids else "error",
        data=run_summary,
    )

    return {
        "run_id": run_id,
        "selected_case_ids": selected_case_ids,
        "passed_case_ids": passed_case_ids,
        "failed_case_ids": failed_case_ids,
        "run_dir": event_writer.run_dir,
        "event_path": event_writer.events_path,
    }


def _expectation_fixtures(case: LoadedCase) -> dict[str, object]:
    fixtures = dict(case.fixtures)
    if "session_blob" in case.fixtures:
        fixtures["downloaded_blob"] = json.dumps(case.fixtures["session_blob"]).encode("utf-8")
    return fixtures


def _write_failure_artifacts(*, artifact_writer: ArtifactWriter, case: LoadedCase, step, result) -> list[str]:
    refs: list[str] = []
    step_dir = artifact_writer.run_dir / "cases" / case.case_id / "steps" / step.step_id
    step_dir.mkdir(parents=True, exist_ok=True)

    if result is not None:
        artifact_writer.write_step_http(
            case_id=case.case_id,
            step_id=step.step_id,
            request=result.request,
            response=result.response,
        )
        if result.request is not None:
            refs.append(str(step_dir / "request.json"))
        if result.response is not None:
            refs.append(str(step_dir / "response.json"))

        payloads = {}
        if "session_blob" in case.fixtures:
            payloads["session_blob"] = case.fixtures["session_blob"]
        if step.operation == "sessions.finalize" and result.request and result.request.body:
            payloads["finalize"] = json.loads(result.request.body.decode("utf-8"))
        if step.operation == "feedback.create" and result.request and result.request.body:
            payloads["feedback"] = json.loads(result.request.body.decode("utf-8"))
        if payloads:
            artifact_writer.write_failure_payloads(case_id=case.case_id, step_id=step.step_id, payloads=payloads)
            for key, filename in {
                "session_blob": "session_blob.json",
                "finalize": "finalize_payload.json",
                "feedback": "feedback_payload.json",
            }.items():
                if key in payloads:
                    refs.append(str(step_dir / filename))

    step_trace_path = step_dir / "step_trace.json"
    step_trace_path.write_text(
        json.dumps(
            {
                "case_id": case.case_id,
                "step_id": step.step_id,
                "operation": step.operation,
                "has_request": bool(result and result.request),
                "has_response": bool(result and result.response),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    refs.append(str(step_trace_path))
    return refs
