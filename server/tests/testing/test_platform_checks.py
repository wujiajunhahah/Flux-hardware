from __future__ import annotations

from pathlib import Path

import pytest

from server.app.testing.target_prep import PreparedShadowTarget
from server.app.testing.transport import TargetProfile
from server.scripts.run_platform_checks import build_parser


def test_run_platform_checks_executes_staging_core_by_default(tmp_path: Path):
    from server.app.testing.platform_checks import run_platform_checks

    calls: list[tuple[str, dict]] = []

    def fake_run_cases(**kwargs):
        calls.append(("run_cases", kwargs))
        return {
            "run_id": "run_staging",
            "selected_case_ids": ["bootstrap_happy_path_v1"],
            "passed_case_ids": ["bootstrap_happy_path_v1"],
            "failed_case_ids": [],
            "run_dir": tmp_path / "artifacts" / "staging",
            "event_path": tmp_path / "artifacts" / "staging" / "events.jsonl",
        }

    summary = run_platform_checks(
        contracts_dir=tmp_path / "contracts",
        cases_root=tmp_path / "cases",
        targets_path=tmp_path / "targets.local.json",
        artifacts_root=tmp_path / "artifacts",
        run_cases_fn=fake_run_cases,
    )

    assert summary["ok"] is True
    assert summary["selected_checks"] == ["staging_core"]
    assert summary["staging_core"]["target"] == "staging"
    assert summary["staging_core"]["ok"] is True
    assert calls[0][1]["target_name"] == "staging"
    assert calls[0][1]["tag"] == "core"


def test_run_platform_checks_can_prepare_and_run_production_shadow(tmp_path: Path):
    from server.app.testing.platform_checks import run_platform_checks

    calls: list[tuple[str, object]] = []

    def fake_run_cases(**kwargs):
        calls.append(("run_cases", kwargs["target_name"]))
        return {
            "run_id": f"run_{kwargs['target_name']}",
            "selected_case_ids": ["bootstrap_shadow_read_v1"],
            "passed_case_ids": ["bootstrap_shadow_read_v1"],
            "failed_case_ids": [],
            "run_dir": tmp_path / "artifacts" / kwargs["target_name"],
            "event_path": tmp_path / "artifacts" / kwargs["target_name"] / "events.jsonl",
        }

    def fake_load_target_profile(targets_path, target_name):
        calls.append(("load_target_profile", target_name))
        return TargetProfile(
            name=target_name,
            base_url="https://api.focux.me",
            allow_writes=False,
            allow_capture=False,
            allow_shadow_read=True,
        )

    def fake_prepare_shadow_target(*, profile, payload):
        calls.append(("prepare_shadow_target", payload["provider_token"]))
        return PreparedShadowTarget(
            auth_action="sign_in",
            user_id="usr_shadow",
            device_id="dev_shadow",
            access_token="token_shadow",
            refresh_token="refresh_shadow",
            expires_in_sec=3600,
        )

    def fake_update_target_profile_auth(*, targets_path, target_name, prepared):
        calls.append(("update_target_profile_auth", target_name, prepared.device_id))
        return {}

    summary = run_platform_checks(
        contracts_dir=tmp_path / "contracts",
        cases_root=tmp_path / "cases",
        targets_path=tmp_path / "targets.local.json",
        artifacts_root=tmp_path / "artifacts",
        run_staging_core=False,
        run_production_shadow=True,
        prepare_production_shadow=True,
        shadow_provider_token="dev:shadow-platform",
        run_cases_fn=fake_run_cases,
        load_target_profile_fn=fake_load_target_profile,
        prepare_shadow_target_fn=fake_prepare_shadow_target,
        update_target_profile_auth_fn=fake_update_target_profile_auth,
    )

    assert summary["ok"] is True
    assert summary["selected_checks"] == ["production_shadow"]
    assert summary["prepared_shadow_target"]["device_id"] == "dev_shadow"
    assert summary["production_shadow"]["target"] == "production"
    assert calls == [
        ("load_target_profile", "production"),
        ("prepare_shadow_target", "dev:shadow-platform"),
        ("update_target_profile_auth", "production", "dev_shadow"),
        ("run_cases", "production"),
    ]


def test_run_platform_checks_requires_provider_token_when_preparing_shadow(tmp_path: Path):
    from server.app.testing.platform_checks import run_platform_checks

    with pytest.raises(ValueError, match="shadow_provider_token"):
        run_platform_checks(
            contracts_dir=tmp_path / "contracts",
            cases_root=tmp_path / "cases",
            targets_path=tmp_path / "targets.local.json",
            artifacts_root=tmp_path / "artifacts",
            run_staging_core=False,
            run_production_shadow=True,
            prepare_production_shadow=True,
        )


def test_platform_checks_parser_supports_skip_staging_and_shadow_flags():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--skip-staging-core",
            "--production-shadow",
            "--prepare-production-shadow",
            "--shadow-provider-token",
            "dev:shadow-platform",
        ]
    )

    assert args.skip_staging_core is True
    assert args.production_shadow is True
    assert args.prepare_production_shadow is True
    assert args.shadow_provider_token == "dev:shadow-platform"
