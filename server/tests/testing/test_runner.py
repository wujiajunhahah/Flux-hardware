from __future__ import annotations

import json
from pathlib import Path

from server.app.testing.runner import run_cases
from server.app.testing.transport import FakeTransport, TransportResponse


def _json_response(payload: dict, status_code: int = 200) -> TransportResponse:
    return TransportResponse(
        status_code=status_code,
        headers={"Content-Type": "application/json"},
        body=json.dumps(payload).encode("utf-8"),
    )


def _write_targets(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "staging": {
                    "base_url": "https://staging.api.focux.me",
                    "allow_writes": True,
                    "allow_capture": True,
                    "allow_shadow_read": False,
                },
                "production": {
                    "base_url": "https://api.focux.me",
                    "allow_writes": False,
                    "allow_capture": False,
                    "allow_shadow_read": True,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_case(path: Path, case_id: str, *, tags: list[str], expect_status: int = 200, second_step: bool = False) -> None:
    steps = [
        {
            "step_id": "sign_up",
            "operation": "auth.sign_up",
            "expect": {
                "ok": True,
                "status": expect_status,
            },
            "extract": {
                "access_token": "$.data.access_token",
                "refresh_token": "$.data.refresh_token",
                "device_id": "$.data.device_id",
            },
        }
    ]
    if second_step:
        steps.append(
            {
                "step_id": "sign_in",
                "operation": "auth.sign_in",
                "expect": {
                    "ok": True,
                    "status": 200,
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "title": case_id,
                "kind": "auth",
                "source": "manual",
                "enabled": True,
                "tags": tags,
                "target_profile": "staging",
                "fixtures": {
                    "provider": "apple",
                    "provider_token": f"dev:{case_id}",
                    "device": {
                        "client_device_key": f"device-{case_id}",
                        "platform": "ios",
                        "device_name": f"Device {case_id}",
                        "app_version": "0.1.0",
                        "os_version": "iOS 18.0",
                    },
                },
                "steps": steps,
                "artifact_policy": {
                    "on_fail": ["http", "step_trace"],
                    "on_pass": ["summary"],
                },
            }
        ),
        encoding="utf-8",
    )


def test_run_cases_filters_by_case_and_tag(tmp_path: Path):
    cases_root = tmp_path / "cases"
    contracts_dir = cases_root / "contracts"
    contracts_dir.mkdir(parents=True)
    targets_path = cases_root / "targets.json"
    _write_targets(targets_path)
    _write_case(contracts_dir / "core_case.json", "core_case", tags=["core"])
    _write_case(contracts_dir / "edge_case.json", "edge_case", tags=["edge"])

    transport = FakeTransport(
        {
            ("POST", "https://staging.api.focux.me/v1/auth/sign-up"): _json_response(
                {
                    "ok": True,
                    "data": {
                        "access_token": "access_01",
                        "refresh_token": "refresh_01",
                        "device_id": "dev_01",
                    },
                }
            )
        }
    )

    case_summary = run_cases(
        contracts_dir=contracts_dir,
        cases_root=cases_root,
        targets_path=targets_path,
        target_name="staging",
        transport=transport,
        case_id="edge_case",
        artifacts_root=tmp_path / "artifacts",
    )
    tag_summary = run_cases(
        contracts_dir=contracts_dir,
        cases_root=cases_root,
        targets_path=targets_path,
        target_name="staging",
        transport=transport,
        tag="core",
        artifacts_root=tmp_path / "artifacts_tag",
    )

    assert case_summary["selected_case_ids"] == ["edge_case"]
    assert tag_summary["selected_case_ids"] == ["core_case"]


def test_runner_stops_a_case_on_failure_and_emits_artifact_refs(tmp_path: Path):
    cases_root = tmp_path / "cases"
    contracts_dir = cases_root / "contracts"
    contracts_dir.mkdir(parents=True)
    targets_path = cases_root / "targets.json"
    _write_targets(targets_path)
    _write_case(contracts_dir / "failing_case.json", "failing_case", tags=["core"], expect_status=200, second_step=True)

    transport = FakeTransport(
        {
            ("POST", "https://staging.api.focux.me/v1/auth/sign-up"): _json_response(
                {
                    "ok": False,
                    "error": {
                        "code": "boom",
                        "message": "expected failure",
                    },
                },
                status_code=500,
            )
        }
    )

    summary = run_cases(
        contracts_dir=contracts_dir,
        cases_root=cases_root,
        targets_path=targets_path,
        target_name="staging",
        transport=transport,
        tag="core",
        artifacts_root=tmp_path / "artifacts",
    )

    events = [json.loads(line) for line in summary["event_path"].read_text(encoding="utf-8").splitlines()]
    failed_event = next(event for event in events if event["type"] == "step_failed")
    assert summary["failed_case_ids"] == ["failing_case"]
    assert len(transport.requests) == 1
    assert failed_event["data"]["artifact_refs"]
    assert events[-1]["type"] == "run_finished"


def test_runner_refuses_mutating_steps_for_production_without_break_glass(tmp_path: Path):
    cases_root = tmp_path / "cases"
    contracts_dir = cases_root / "contracts"
    contracts_dir.mkdir(parents=True)
    targets_path = cases_root / "targets.json"
    _write_targets(targets_path)
    _write_case(contracts_dir / "prod_case.json", "prod_case", tags=["core"])

    transport = FakeTransport()

    summary = run_cases(
        contracts_dir=contracts_dir,
        cases_root=cases_root,
        targets_path=targets_path,
        target_name="production",
        transport=transport,
        tag="core",
        artifacts_root=tmp_path / "artifacts",
    )

    assert summary["failed_case_ids"] == ["prod_case"]
    assert transport.requests == []


def test_runner_writes_case_summary_and_artifact_written_events(tmp_path: Path):
    cases_root = tmp_path / "cases"
    contracts_dir = cases_root / "contracts"
    contracts_dir.mkdir(parents=True)
    targets_path = cases_root / "targets.json"
    _write_targets(targets_path)
    _write_case(contracts_dir / "summary_case.json", "summary_case", tags=["core"])

    transport = FakeTransport(
        {
            ("POST", "https://staging.api.focux.me/v1/auth/sign-up"): _json_response(
                {
                    "ok": True,
                    "data": {
                        "access_token": "access_01",
                        "refresh_token": "refresh_01",
                        "device_id": "dev_01",
                    },
                }
            )
        }
    )

    summary = run_cases(
        contracts_dir=contracts_dir,
        cases_root=cases_root,
        targets_path=targets_path,
        target_name="staging",
        transport=transport,
        tag="core",
        artifacts_root=tmp_path / "artifacts",
    )

    case_summary_path = (
        tmp_path
        / "artifacts"
        / "contracts"
        / summary["run_id"]
        / "cases"
        / "summary_case"
        / "summary.json"
    )
    run_summary_path = tmp_path / "artifacts" / "contracts" / summary["run_id"] / "run.json"
    assert case_summary_path.exists()
    assert run_summary_path.exists()

    events = [json.loads(line) for line in summary["event_path"].read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "artifact_written" for event in events)
