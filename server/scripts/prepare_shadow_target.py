from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..app.testing.target_prep import (
    build_shadow_auth_payload,
    load_target_profile,
    prepare_shadow_target,
    update_target_profile_auth,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    default_targets_path = repo_root / "server" / "cases" / "targets.local.json"

    parser = argparse.ArgumentParser(description="Prepare a shadow target profile with fresh auth material")
    parser.add_argument("--target", default="production", help="target profile name to update")
    parser.add_argument("--targets-path", type=Path, default=default_targets_path)
    parser.add_argument("--provider", default="apple")
    parser.add_argument("--provider-token", required=True)
    parser.add_argument("--client-device-key", default="shadow-device-production")
    parser.add_argument("--platform", default="ios")
    parser.add_argument("--device-name", default="Shadow Readonly Device")
    parser.add_argument("--app-version", default="0.1.0")
    parser.add_argument("--os-version", default="iOS 18.0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    profile = load_target_profile(args.targets_path, args.target)
    payload = build_shadow_auth_payload(
        provider=args.provider,
        provider_token=args.provider_token,
        client_device_key=args.client_device_key,
        platform=args.platform,
        device_name=args.device_name,
        app_version=args.app_version,
        os_version=args.os_version,
    )
    prepared = prepare_shadow_target(profile=profile, payload=payload)
    update_target_profile_auth(
        targets_path=args.targets_path,
        target_name=args.target,
        prepared=prepared,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "target": args.target,
                "targets_path": str(args.targets_path),
                "base_url": profile.base_url,
                "auth_action": prepared.auth_action,
                "user_id": prepared.user_id,
                "device_id": prepared.device_id,
                "expires_in_sec": prepared.expires_in_sec,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
