from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..app.testing.platform_checks import run_platform_checks


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    default_cases_root = repo_root / "server" / "cases"

    parser = argparse.ArgumentParser(description="Run aggregated FluxChi platform checks")
    parser.add_argument("--targets-path", type=Path, default=default_cases_root / "targets.local.json")
    parser.add_argument("--cases-root", type=Path, default=default_cases_root)
    parser.add_argument("--contracts-dir", type=Path, default=default_cases_root / "contracts")
    parser.add_argument("--artifacts-root", type=Path, default=repo_root / "artifacts")
    parser.add_argument("--staging-target", default="staging")
    parser.add_argument("--production-target", default="production")
    parser.add_argument("--skip-staging-core", action="store_true")
    parser.add_argument("--production-shadow", action="store_true")
    parser.add_argument("--prepare-production-shadow", action="store_true")
    parser.add_argument("--shadow-provider", default="apple")
    parser.add_argument("--shadow-provider-token")
    parser.add_argument("--shadow-client-device-key", default="shadow-device-production")
    parser.add_argument("--shadow-platform", default="ios")
    parser.add_argument("--shadow-device-name", default="Shadow Readonly Device")
    parser.add_argument("--shadow-app-version", default="0.1.0")
    parser.add_argument("--shadow-os-version", default="iOS 18.0")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.prepare_production_shadow and not args.production_shadow:
        parser.error("--prepare-production-shadow requires --production-shadow")
    if args.prepare_production_shadow and not args.shadow_provider_token:
        parser.error("--shadow-provider-token is required with --prepare-production-shadow")

    summary = run_platform_checks(
        contracts_dir=args.contracts_dir,
        cases_root=args.cases_root,
        targets_path=args.targets_path,
        artifacts_root=args.artifacts_root,
        staging_target=args.staging_target,
        production_target=args.production_target,
        run_staging_core=not args.skip_staging_core,
        run_production_shadow=args.production_shadow,
        prepare_production_shadow=args.prepare_production_shadow,
        shadow_provider=args.shadow_provider,
        shadow_provider_token=args.shadow_provider_token,
        shadow_client_device_key=args.shadow_client_device_key,
        shadow_platform=args.shadow_platform,
        shadow_device_name=args.shadow_device_name,
        shadow_app_version=args.shadow_app_version,
        shadow_os_version=args.shadow_os_version,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
