from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import RunEvent


class RunEventWriter:
    def __init__(self, *, artifacts_root: Path, run_id: str):
        self.run_id = run_id
        self._seq = 0
        self.run_dir = Path(artifacts_root) / "contracts" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"

    def write(
        self,
        *,
        type: str,
        status: str,
        case_id: str | None = None,
        step_id: str | None = None,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> RunEvent:
        self._seq += 1
        event = RunEvent(
            schema_version=1,
            seq=self._seq,
            run_id=self.run_id,
            case_id=case_id,
            step_id=step_id,
            type=type,
            status=status,
            message=message,
            data=data or {},
        )
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json())
            handle.write("\n")
        return event
