from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Iterable

from server.app.core.db import (
    DatabaseConfigurationError,
    DatabaseDependencyError,
    connect,
)
from server.app.core.settings import SERVER_ROOT, settings

TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename TEXT PRIMARY KEY,
    checksum_sha256 TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply SQL migrations for FluxChi platform backend.")
    parser.add_argument(
        "--migrations-dir",
        default=str(SERVER_ROOT / "migrations"),
        help="Directory containing ordered .sql migration files.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Optional Postgres DSN. Falls back to FLUX_DATABASE_URL.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show applied and pending migrations without executing them.",
    )
    return parser.parse_args()


def discover_migrations(migrations_dir: Path) -> list[Path]:
    if not migrations_dir.exists():
        raise FileNotFoundError(f"migrations directory does not exist: {migrations_dir}")
    return sorted(path for path in migrations_dir.iterdir() if path.suffix == ".sql")


def read_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_sql(sql: str) -> str:
    normalized = sql.lstrip()
    if normalized.startswith("BEGIN;"):
        normalized = normalized[len("BEGIN;") :].lstrip()
    stripped = normalized.rstrip()
    if stripped.endswith("COMMIT;"):
        stripped = stripped[: -len("COMMIT;")].rstrip()
    return stripped + "\n"


def ensure_tracking_table(conn: object) -> None:
    with conn.cursor() as cur:
        cur.execute(TRACKING_TABLE_SQL)
    conn.commit()


def load_applied_migrations(conn: object) -> dict[str, tuple[str, object]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT filename, checksum_sha256, applied_at FROM schema_migrations ORDER BY filename"
        )
        rows = cur.fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def format_status(paths: Iterable[Path], applied: dict[str, tuple[str, object]]) -> str:
    lines: list[str] = []
    for path in paths:
        checksum = read_checksum(path)
        if path.name in applied:
            applied_checksum, applied_at = applied[path.name]
            marker = "applied"
            if applied_checksum != checksum:
                marker = "checksum-mismatch"
            lines.append(f"{marker:18} {path.name} {applied_at}")
        else:
            lines.append(f"pending            {path.name}")
    return "\n".join(lines)


def apply_migration(conn: object, path: Path, checksum: str) -> None:
    sql = normalize_sql(path.read_text(encoding="utf-8"))
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute(
                "INSERT INTO schema_migrations (filename, checksum_sha256) VALUES (%s, %s)",
                (path.name, checksum),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def main() -> int:
    args = parse_args()
    migrations_dir = Path(args.migrations_dir).resolve()

    try:
        migrations = discover_migrations(migrations_dir)
        conn = connect(database_url=args.database_url)
    except (FileNotFoundError, DatabaseConfigurationError, DatabaseDependencyError) as exc:
        print(f"migration bootstrap failed: {exc}", file=sys.stderr)
        return 1

    try:
        ensure_tracking_table(conn)
        applied = load_applied_migrations(conn)
        print(
            f"database={settings.database_url or '<unset>'} migrations_dir={migrations_dir}",
            file=sys.stderr,
        )

        if args.status:
            print(format_status(migrations, applied))
            return 0

        for path in migrations:
            checksum = read_checksum(path)
            if path.name in applied:
                applied_checksum, _ = applied[path.name]
                if applied_checksum != checksum:
                    print(
                        f"checksum mismatch for already applied migration: {path.name}",
                        file=sys.stderr,
                    )
                    return 1
                print(f"skip {path.name}")
                continue

            apply_migration(conn, path, checksum)
            print(f"apply {path.name}")

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
