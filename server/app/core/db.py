from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .settings import settings


class DatabaseConfigurationError(RuntimeError):
    pass


class DatabaseDependencyError(RuntimeError):
    pass


def import_psycopg() -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - dependency may be absent during bootstrap
        raise DatabaseDependencyError(
            "psycopg is required for database access. Install requirements and set FLUX_DATABASE_URL."
        ) from exc
    return psycopg


def resolve_database_url(override: str | None = None) -> str:
    database_url = override or settings.database_url
    if not database_url:
        raise DatabaseConfigurationError("FLUX_DATABASE_URL is not set")
    return database_url


def connect(database_url: str | None = None, **kwargs: Any) -> Any:
    # V1 intentionally uses sync psycopg with FastAPI `def` routes. This keeps
    # repository code simple until traffic or p95 latency justifies an async move.
    psycopg = import_psycopg()
    kwargs.setdefault("row_factory", psycopg.rows.dict_row)
    return psycopg.connect(resolve_database_url(database_url), **kwargs)


@contextmanager
def connection(database_url: str | None = None, **kwargs: Any) -> Iterator[Any]:
    conn = connect(database_url=database_url, **kwargs)
    try:
        yield conn
    finally:
        conn.close()
