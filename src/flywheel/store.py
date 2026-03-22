"""
FluxChi -- Flywheel Data Store
=========================================================
本地 SQLite 数据存储, 用于:
  1. 记录每 N 秒的 ModuleReading 快照 (训练特征)
  2. 记录用户标注事件 ("我很累" / "我很清醒" / KSS)
  3. 记录自动推断标签 (休息前5min = fatigued)
  4. 导出为训练数据集

表结构:
  snapshots: 定期快照 (所有模块的 dimensions + quality)
  labels:    标注事件 (用户主动 + 系统推断)
  sessions:  会话元数据

隐私:
  - 仅存储 dimensions 数值, 不存储原始视频/EMG 波形
  - 数据完全本地, 不上传
=========================================================
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..sensor_module import ModuleReading


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  数据结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class LabelEvent:
    """标注事件。"""
    timestamp: float
    label: str              # "alert" / "fatigued" / "drowsy"
    source: str             # "user_button" / "user_kss" / "auto_pre_break" / "auto_morning"
    kss: Optional[int]      # Karolinska Sleepiness Scale 1-9, 可选
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "source": self.source,
            "kss": self.kss,
            "note": self.note,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  存储引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FlywheelStore:
    """本地 SQLite 飞轮数据存储。

    用法:
        store = FlywheelStore("~/.fluxchi/flywheel.db")
        store.save_snapshot(readings, fused_stamina=85.0)
        store.save_label(LabelEvent(...))
        dataset = store.export_training_data()
    """

    SNAPSHOT_INTERVAL_SEC = 5.0  # 快照间隔

    def __init__(self, db_path: str = "~/.fluxchi/flywheel.db") -> None:
        self._db_path = Path(os.path.expanduser(db_path))
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        self._last_snapshot: float = 0.0

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                session_id TEXT,
                fused_stamina REAL,
                fused_state TEXT,
                modules_json TEXT NOT NULL,
                alerts_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_snap_ts ON snapshots(timestamp);

            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                label TEXT NOT NULL,
                source TEXT NOT NULL,
                kss INTEGER,
                note TEXT,
                window_start REAL,
                window_end REAL
            );
            CREATE INDEX IF NOT EXISTS idx_label_ts ON labels(timestamp);

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_time REAL NOT NULL,
                end_time REAL,
                snapshot_count INTEGER DEFAULT 0,
                label_count INTEGER DEFAULT 0,
                metadata_json TEXT
            );
        """)
        self._conn.commit()

    # ── 快照 ──────────────────────────────────────────────

    def save_snapshot(
        self,
        readings: Dict[str, ModuleReading],
        fused_stamina: Optional[float] = None,
        fused_state: Optional[str] = None,
        alerts: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """保存一条快照。默认按 SNAPSHOT_INTERVAL_SEC 节流。

        返回: 是否实际保存了 (可能被节流跳过)
        """
        now = time.time()
        if not force and (now - self._last_snapshot) < self.SNAPSHOT_INTERVAL_SEC:
            return False

        modules_dict = {}
        for mid, r in readings.items():
            modules_dict[mid] = {
                "stamina": round(r.stamina, 1),
                "quality": round(r.quality, 2),
                "dimensions": {k: round(v, 3) if isinstance(v, float) else v
                               for k, v in r.dimensions.items()},
            }

        self._conn.execute(
            """INSERT INTO snapshots
               (timestamp, session_id, fused_stamina, fused_state, modules_json, alerts_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                now,
                session_id,
                round(fused_stamina, 1) if fused_stamina is not None else None,
                fused_state,
                json.dumps(modules_dict, ensure_ascii=False),
                json.dumps(alerts or [], ensure_ascii=False),
            ),
        )
        self._conn.commit()
        self._last_snapshot = now
        return True

    # ── 标注 ──────────────────────────────────────────────

    def save_label(
        self,
        event: LabelEvent,
        window_sec: float = 300.0,
    ) -> None:
        """保存一条标注事件。

        window_sec: 标注覆盖的时间窗口 (默认 5 分钟)。
            点击"我很累"时, 标记 [now-window_sec, now] 的快照。
        """
        self._conn.execute(
            """INSERT INTO labels
               (timestamp, label, source, kss, note, window_start, window_end)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event.timestamp,
                event.label,
                event.source,
                event.kss,
                event.note,
                event.timestamp - window_sec,
                event.timestamp,
            ),
        )
        self._conn.commit()

    # ── 导出训练数据 ──────────────────────────────────────

    def export_training_data(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """导出带标签的训练数据。

        逻辑: 对每条 snapshot, 找最近的 label (其 window 覆盖该 snapshot),
        组合为 {features: ..., label: ..., kss: ...}。

        返回: list of dicts, 每条是一个训练样本。
        """
        query = "SELECT timestamp, modules_json, fused_stamina FROM snapshots"
        conditions = []
        params: List[Any] = []
        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp"

        rows = self._conn.execute(query, params).fetchall()

        # 预加载所有 labels
        labels = self._conn.execute(
            "SELECT window_start, window_end, label, kss FROM labels ORDER BY timestamp"
        ).fetchall()

        samples = []
        for ts, modules_json, fused_stamina in rows:
            modules = json.loads(modules_json)

            # 找覆盖此时间点的标签
            matched_label = None
            matched_kss = None
            for ws, we, label, kss in labels:
                if ws <= ts <= we:
                    matched_label = label
                    matched_kss = kss

            if matched_label is None:
                continue  # 无标签的快照跳过

            # 展平特征
            features: Dict[str, float] = {}
            for mid, mdata in modules.items():
                features[f"{mid}_stamina"] = mdata.get("stamina", 0)
                features[f"{mid}_quality"] = mdata.get("quality", 0)
                for dim_k, dim_v in mdata.get("dimensions", {}).items():
                    features[f"{mid}_{dim_k}"] = dim_v

            samples.append({
                "timestamp": ts,
                "features": features,
                "fused_stamina": fused_stamina,
                "label": matched_label,
                "kss": matched_kss,
            })

        return samples

    # ── 统计 ──────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        snap_count = self._conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        label_count = self._conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        labeled_snaps = len(self.export_training_data())
        return {
            "db_path": str(self._db_path),
            "total_snapshots": snap_count,
            "total_labels": label_count,
            "labeled_samples": labeled_snaps,
        }

    def close(self) -> None:
        self._conn.close()
