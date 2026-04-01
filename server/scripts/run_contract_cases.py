from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..app.testing.runner import run_cases


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    default_cases_root = repo_root / "server" / "cases"

    parser = argparse.ArgumentParser(description="Run FluxChi contract cases")
    parser.add_argument("--target", default="staging", help="target profile name from targets file")
    parser.add_argument("--case", dest="case_id", help="single case_id to execute")
    parser.add_argument("--tag", help="run only cases with a given tag")
    parser.add_argument("--break-glass", action="store_true", help="allow mutating production operations")
    parser.add_argument("--cases-root", type=Path, default=default_cases_root)
    parser.add_argument("--contracts-dir", type=Path, default=default_cases_root / "contracts")
    parser.add_argument("--targets-path", type=Path, default=default_cases_root / "targets.example.json")
    parser.add_argument("--artifacts-root", type=Path, default=repo_root / "artifacts")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_cases(
        contracts_dir=args.contracts_dir,
        cases_root=args.cases_root,
        targets_path=args.targets_path,
        target_name=args.target,
        case_id=args.case_id,
        tag=args.tag,
        break_glass=args.break_glass,
        artifacts_root=args.artifacts_root,
    )
    print(
        json.dumps(
            {
                **summary,
                "run_dir": str(summary["run_dir"]),
                "event_path": str(summary["event_path"]),
            },
            indent=2,
        )
    )
    return 0 if not summary["failed_case_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
