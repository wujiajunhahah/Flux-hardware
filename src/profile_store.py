"""Lightweight personalization profile storage."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROFILE_PATH = Path(os.path.expanduser("~/.fluxchi/personalization_profile.json"))


def profile_path() -> Path:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return PROFILE_PATH


def load_profile() -> Optional[Dict[str, Any]]:
    path = profile_path()
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def save_profile(profile: Dict[str, Any]) -> None:
    path = profile_path()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def normalize_updated_at(value: str) -> str:
    dt = _parse_updated_at(value)
    return dt.isoformat().replace("+00:00", "Z")


def should_replace_profile(
    incoming: Dict[str, Any],
    current: Optional[Dict[str, Any]],
) -> bool:
    if current is None:
        return True

    current_updated_at = str(current.get("updatedAt") or "").strip()
    if not current_updated_at:
        return True

    incoming_dt = _parse_updated_at(str(incoming.get("updatedAt") or ""))
    current_dt = _parse_updated_at(current_updated_at)
    return incoming_dt >= current_dt


def _parse_updated_at(value: str) -> datetime:
    raw = value.strip()
    if not raw:
        raise ValueError("updatedAt is required")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
