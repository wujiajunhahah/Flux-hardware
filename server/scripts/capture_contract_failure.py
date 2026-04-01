from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from ..app.testing.redaction import redact_payload


def capture_failure_bundle(bundle: Mapping[str, Any], *, output_dir: str | Path) -> Path:
    _validate_bundle(bundle)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    redacted_bundle = redact_payload(bundle)
    captured_case = {
        "case_id": bundle["case_id"],
        "title": bundle["title"],
        "kind": bundle["kind"],
        "source": "captured",
        "enabled": False,
        "target_profile": bundle.get("target_profile", "staging"),
        "tags": _captured_tags(bundle.get("tags", [])),
        "seed": bundle.get("seed", {}),
        "fixtures": {
            "failure_bundle": redacted_bundle,
        },
        "steps": [
            {
                "step_id": step["step_id"],
                "operation": step["operation"],
                "input": redact_payload(step.get("input", {})),
                "expect": step.get("expect", {}),
                "extract": step.get("extract", {}),
            }
            for step in bundle["steps"]
        ],
    }

    case_path = output_path / f"{bundle['case_id']}.json"
    case_path.write_text(json.dumps(captured_case, ensure_ascii=False, indent=2), encoding="utf-8")
    return case_path


def _captured_tags(tags: list[str]) -> list[str]:
    values = list(tags)
    if "captured" not in values:
        values.append("captured")
    return values


def _validate_bundle(bundle: Mapping[str, Any]) -> None:
    required = ("case_id", "title", "kind", "steps")
    if not all(key in bundle for key in required):
        raise ValueError("malformed failure bundle: missing required fields")
    if not isinstance(bundle["steps"], list) or not bundle["steps"]:
        raise ValueError("malformed failure bundle: steps must be a non-empty list")
    for step in bundle["steps"]:
        if not isinstance(step, Mapping):
            raise ValueError("malformed failure bundle: step must be an object")
        if "step_id" not in step or "operation" not in step:
            raise ValueError("malformed failure bundle: step is missing identifiers")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a sanitized contract failure bundle")
    parser.add_argument("--input", type=Path, required=True, help="path to sanitized failure bundle JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for captured case JSON files")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle = json.loads(args.input.read_text(encoding="utf-8"))
    case_path = capture_failure_bundle(bundle, output_dir=args.output_dir)
    print(case_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
