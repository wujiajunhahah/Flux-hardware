from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import LoadedCase


def load_cases(
    contracts_dir: str | Path,
    *,
    cases_root: str | Path | None = None,
    case_id: str | None = None,
    tag: str | None = None,
    include_disabled: bool = False,
) -> list[LoadedCase]:
    contracts_path = Path(contracts_dir)
    root = Path(cases_root) if cases_root is not None else contracts_path.parent

    loaded: list[LoadedCase] = []
    for case_file in sorted(contracts_path.glob("*.json")):
        payload = json.loads(case_file.read_text(encoding="utf-8"))
        seed = payload.get("seed", {})
        payload = _expand_templates(payload, seed)
        payload = _resolve_fixture_refs(payload, root)
        case = LoadedCase.model_validate(payload)

        if not include_disabled and not case.enabled:
            continue
        if case_id is not None and case.case_id != case_id:
            continue
        if tag is not None and tag not in case.tags:
            continue
        loaded.append(case)

    return loaded


def _expand_templates(value: Any, seed: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        expanded: dict[str, Any] = {}
        for key, item in value.items():
            target_key = key[:-9] if key.endswith("_template") else key
            expanded[target_key] = _expand_templates(item, seed)
        return expanded
    if isinstance(value, list):
        return [_expand_templates(item, seed) for item in value]
    if isinstance(value, str) and "{" in value:
        return value.format_map(seed)
    return value


def _resolve_fixture_refs(value: Any, cases_root: Path) -> Any:
    if isinstance(value, dict):
        resolved: dict[str, Any] = {}
        for key, item in value.items():
            if key.endswith("_ref") and isinstance(item, str) and _looks_like_fixture_path(item):
                fixture_path = _resolve_fixture_path(cases_root, item)
                target_key = key[:-4]
                resolved[target_key] = json.loads(fixture_path.read_text(encoding="utf-8"))
                continue
            resolved[key] = _resolve_fixture_refs(item, cases_root)
        return resolved
    if isinstance(value, list):
        return [_resolve_fixture_refs(item, cases_root) for item in value]
    return value


def _resolve_fixture_path(cases_root: Path, relative_path: str) -> Path:
    candidate = (cases_root / relative_path).resolve()
    root = cases_root.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"fixture path escapes cases root: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"fixture not found: {relative_path}")
    return candidate


def _looks_like_fixture_path(value: str) -> bool:
    return "/" in value or value.endswith(".json")
