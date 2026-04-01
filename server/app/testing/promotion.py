from __future__ import annotations

import shutil
from pathlib import Path


def promote_captured_case(captured_case_path: str | Path, contracts_dir: str | Path) -> Path:
    source = Path(captured_case_path)
    target_dir = Path(contracts_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    return target
