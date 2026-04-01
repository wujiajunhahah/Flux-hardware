from __future__ import annotations

import json
from pathlib import Path

from server.app.testing.case_loader import load_cases


def _repo_cases_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "cases"


def _base_case_payload(case_id: str, *, enabled: bool = True, tags: list[str] | None = None) -> dict:
    return {
        "case_id": case_id,
        "title": case_id,
        "kind": "session_upload",
        "source": "manual",
        "enabled": enabled,
        "tags": tags or [],
        "target_profile": "staging",
        "steps": [
            {
                "step_id": "create_session",
                "operation": "sessions.create",
                "input": {},
            }
        ],
    }


def test_loader_resolves_fixture_references_and_seed_templates_from_repo_cases():
    cases = load_cases(_repo_cases_dir() / "contracts", cases_root=_repo_cases_dir(), case_id="session_upload_happy_path_v1")

    assert len(cases) == 1
    case = cases[0]
    assert case.fixtures["provider_token"] == "dev:happy01"
    assert case.fixtures["device"]["client_device_key"] == "case-device-happy01"
    assert case.fixtures["device"]["device_name"] == "Contract Case Device happy01"
    assert case.fixtures["session_blob"]["session"]["title"] == "Minimal Session Blob"


def test_loader_filters_by_enabled_case_id_and_tag(tmp_path: Path):
    cases_root = tmp_path / "cases"
    contracts_dir = cases_root / "contracts"
    contracts_dir.mkdir(parents=True)

    (contracts_dir / "core_enabled.json").write_text(
        json.dumps(_base_case_payload("core_enabled", enabled=True, tags=["core"])),
        encoding="utf-8",
    )
    (contracts_dir / "core_disabled.json").write_text(
        json.dumps(_base_case_payload("core_disabled", enabled=False, tags=["core"])),
        encoding="utf-8",
    )
    (contracts_dir / "edge_enabled.json").write_text(
        json.dumps(_base_case_payload("edge_enabled", enabled=True, tags=["edge"])),
        encoding="utf-8",
    )

    default_cases = load_cases(contracts_dir, cases_root=cases_root)
    filtered_by_id = load_cases(contracts_dir, cases_root=cases_root, case_id="edge_enabled")
    filtered_by_tag = load_cases(contracts_dir, cases_root=cases_root, tag="core")

    assert [case.case_id for case in default_cases] == ["core_enabled", "edge_enabled"]
    assert [case.case_id for case in filtered_by_id] == ["edge_enabled"]
    assert [case.case_id for case in filtered_by_tag] == ["core_enabled"]
