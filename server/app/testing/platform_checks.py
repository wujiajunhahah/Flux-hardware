from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .runner import run_cases
from .target_prep import (
    build_shadow_auth_payload,
    load_target_profile,
    prepare_shadow_target,
    update_target_profile_auth,
)


def run_platform_checks(
    *,
    contracts_dir: str | Path,
    cases_root: str | Path,
    targets_path: str | Path,
    artifacts_root: str | Path = "artifacts",
    staging_target: str = "staging",
    production_target: str = "production",
    run_staging_core: bool = True,
    run_production_shadow: bool = False,
    prepare_production_shadow: bool = False,
    shadow_provider: str = "apple",
    shadow_provider_token: str | None = None,
    shadow_client_device_key: str = "shadow-device-production",
    shadow_platform: str = "ios",
    shadow_device_name: str = "Shadow Readonly Device",
    shadow_app_version: str = "0.1.0",
    shadow_os_version: str = "iOS 18.0",
    run_cases_fn: Callable[..., dict[str, Any]] = run_cases,
    load_target_profile_fn: Callable[..., Any] = load_target_profile,
    prepare_shadow_target_fn: Callable[..., Any] = prepare_shadow_target,
    update_target_profile_auth_fn: Callable[..., Any] = update_target_profile_auth,
) -> dict[str, Any]:
    if not run_staging_core and not run_production_shadow:
        raise ValueError("at least one platform check must be selected")
    if prepare_production_shadow and not run_production_shadow:
        raise ValueError("prepare_production_shadow requires run_production_shadow")
    if prepare_production_shadow and not shadow_provider_token:
        raise ValueError("shadow_provider_token is required when prepare_production_shadow is enabled")

    summary: dict[str, Any] = {
        "ok": True,
        "selected_checks": [],
    }

    if run_staging_core:
        staging_summary = run_cases_fn(
            contracts_dir=contracts_dir,
            cases_root=cases_root,
            targets_path=targets_path,
            target_name=staging_target,
            tag="core",
            artifacts_root=artifacts_root,
        )
        summary["selected_checks"].append("staging_core")
        summary["staging_core"] = {
            "ok": not staging_summary["failed_case_ids"],
            "target": staging_target,
            **_serialize_run_summary(staging_summary),
        }
        summary["ok"] = summary["ok"] and summary["staging_core"]["ok"]

    if run_production_shadow:
        if prepare_production_shadow:
            profile = load_target_profile_fn(targets_path, production_target)
            payload = build_shadow_auth_payload(
                provider=shadow_provider,
                provider_token=shadow_provider_token or "",
                client_device_key=shadow_client_device_key,
                platform=shadow_platform,
                device_name=shadow_device_name,
                app_version=shadow_app_version,
                os_version=shadow_os_version,
            )
            prepared = prepare_shadow_target_fn(profile=profile, payload=payload)
            update_target_profile_auth_fn(
                targets_path=targets_path,
                target_name=production_target,
                prepared=prepared,
            )
            summary["prepared_shadow_target"] = {
                "auth_action": prepared.auth_action,
                "user_id": prepared.user_id,
                "device_id": prepared.device_id,
                "expires_in_sec": prepared.expires_in_sec,
            }

        production_summary = run_cases_fn(
            contracts_dir=contracts_dir,
            cases_root=cases_root,
            targets_path=targets_path,
            target_name=production_target,
            tag="shadow",
            artifacts_root=artifacts_root,
        )
        summary["selected_checks"].append("production_shadow")
        summary["production_shadow"] = {
            "ok": not production_summary["failed_case_ids"],
            "target": production_target,
            **_serialize_run_summary(production_summary),
        }
        summary["ok"] = summary["ok"] and summary["production_shadow"]["ok"]

    return summary


def _serialize_run_summary(run_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_summary["run_id"],
        "selected_case_ids": list(run_summary["selected_case_ids"]),
        "passed_case_ids": list(run_summary["passed_case_ids"]),
        "failed_case_ids": list(run_summary["failed_case_ids"]),
        "run_dir": str(run_summary["run_dir"]),
        "event_path": str(run_summary["event_path"]),
    }
