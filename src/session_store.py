"""会话归档：落盘到仓库根下 data/sessions/（目录已在 .gitignore）。"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = REPO_ROOT / "data" / "sessions"

_SAFE_ID = re.compile(r"^[a-f0-9\-]{8,64}$", re.I)


def ensure_sessions_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def archive_filename(ts_compact: str, session_id: str) -> str:
    """与 iOS `ExportManager` 一致：`fluxchi_{yyyyMMdd_HHmmss}_{uuid}.json`。"""
    return f"fluxchi_{ts_compact}_{session_id}.json"


def _session_id_from_package(data: Dict[str, Any]) -> str:
    s = data.get("session")
    if isinstance(s, dict):
        sid = (s.get("id") or "").strip()
        if sid and _SAFE_ID.match(sid):
            return sid
    return str(uuid.uuid4())


def _remove_previous_archives(session_id: str) -> None:
    """同一 session id 只保留最新一份文件（避免多次 ingest 堆积）。"""
    for p in SESSIONS_DIR.glob(f"fluxchi_*_{session_id}.json"):
        try:
            p.unlink()
        except OSError:
            pass
    leg = SESSIONS_DIR / f"{session_id}.json"
    if leg.is_file():
        try:
            leg.unlink()
        except OSError:
            pass


def find_session_path(session_id: str) -> Optional[Path]:
    if not _SAFE_ID.match(session_id.strip()):
        return None
    sid = session_id.strip()
    matches = list(SESSIONS_DIR.glob(f"fluxchi_*_{sid}.json"))
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)
    leg = SESSIONS_DIR / f"{sid}.json"
    if leg.is_file():
        return leg
    return None


def save_session_package(data: Dict[str, Any]) -> str:
    ensure_sessions_dir()
    sid = _session_id_from_package(data)
    if isinstance(data.get("session"), dict):
        data["session"]["id"] = sid

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    fname = archive_filename(ts, sid)
    data["suggestedArchiveFileName"] = fname

    _remove_previous_archives(sid)
    path = SESSIONS_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return sid


def list_sessions_meta() -> List[Dict[str, Any]]:
    ensure_sessions_dir()
    out: List[Dict[str, Any]] = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            s = d.get("session") or {}
            out.append(
                {
                    "id": s.get("id", p.stem),
                    "title": s.get("title", ""),
                    "startedAt": s.get("startedAt", ""),
                    "source": s.get("source", ""),
                    "durationSec": s.get("durationSec", 0),
                    "snapshotCount": (d.get("signalMeta") or {}).get("snapshotCount", 0),
                    "archiveFile": p.name,
                    "suggestedArchiveFileName": d.get("suggestedArchiveFileName") or p.name,
                }
            )
        except Exception:
            continue
    return out


def load_session(session_id: str) -> Dict[str, Any]:
    if not _SAFE_ID.match(session_id.strip()):
        raise ValueError("invalid session id")
    path = find_session_path(session_id.strip())
    if path is None or not path.is_file():
        raise FileNotFoundError(session_id)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def session_file_path(session_id: str) -> Path:
    if not _SAFE_ID.match(session_id.strip()):
        raise ValueError("invalid session id")
    path = find_session_path(session_id.strip())
    if path is None or not path.is_file():
        raise FileNotFoundError(session_id)
    return path
