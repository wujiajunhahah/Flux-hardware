from __future__ import annotations

from datetime import datetime

SESSION_COLUMNS = """
    session_id,
    user_id,
    device_id,
    idempotency_key,
    status,
    source,
    title,
    started_at,
    ended_at,
    duration_sec,
    snapshot_count,
    schema_version,
    content_type,
    blob_object_key,
    blob_size_bytes,
    blob_sha256,
    created_at,
    updated_at
"""


class SessionRepository:
    def get_session(self, conn: object, user_id: str, session_id: str, *, for_update: bool = False) -> dict | None:
        sql = f"SELECT {SESSION_COLUMNS} FROM sessions WHERE user_id = %s AND session_id = %s"
        if for_update:
            sql += " FOR UPDATE"
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, session_id))
            return cur.fetchone()

    def get_session_any(self, conn: object, session_id: str, *, for_update: bool = False) -> dict | None:
        sql = f"SELECT {SESSION_COLUMNS} FROM sessions WHERE session_id = %s"
        if for_update:
            sql += " FOR UPDATE"
        with conn.cursor() as cur:
            cur.execute(sql, (session_id,))
            return cur.fetchone()

    def get_session_by_idempotency(self, conn: object, user_id: str, idempotency_key: str) -> dict | None:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {SESSION_COLUMNS} FROM sessions WHERE user_id = %s AND idempotency_key = %s",
                (user_id, idempotency_key),
            )
            return cur.fetchone()

    def create_session(
        self,
        conn: object,
        *,
        session_id: str,
        user_id: str,
        device_id: str,
        idempotency_key: str,
        status: str,
        source: str,
        title: str | None,
        started_at: datetime,
        ended_at: datetime,
        duration_sec: int,
        snapshot_count: int,
        schema_version: int,
        content_type: str,
        blob_object_key: str,
        blob_size_bytes: int,
        blob_sha256: str,
        now: datetime,
    ) -> dict:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO sessions (
                    session_id,
                    user_id,
                    device_id,
                    idempotency_key,
                    status,
                    source,
                    title,
                    started_at,
                    ended_at,
                    duration_sec,
                    snapshot_count,
                    schema_version,
                    content_type,
                    blob_object_key,
                    blob_size_bytes,
                    blob_sha256,
                    created_at,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {SESSION_COLUMNS}
                """,
                (
                    session_id,
                    user_id,
                    device_id,
                    idempotency_key,
                    status,
                    source,
                    title,
                    started_at,
                    ended_at,
                    duration_sec,
                    snapshot_count,
                    schema_version,
                    content_type,
                    blob_object_key,
                    blob_size_bytes,
                    blob_sha256,
                    now,
                    now,
                ),
            )
            return cur.fetchone()

    def update_session_ready(
        self,
        conn: object,
        *,
        user_id: str,
        session_id: str,
        blob_object_key: str,
        blob_size_bytes: int,
        blob_sha256: str,
        updated_at: datetime,
    ) -> dict:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE sessions
                SET status = 'ready',
                    blob_object_key = %s,
                    blob_size_bytes = %s,
                    blob_sha256 = %s,
                    updated_at = %s
                WHERE user_id = %s AND session_id = %s
                RETURNING {SESSION_COLUMNS}
                """,
                (
                    blob_object_key,
                    blob_size_bytes,
                    blob_sha256,
                    updated_at,
                    user_id,
                    session_id,
                ),
            )
            return cur.fetchone()

    def list_sessions(
        self,
        conn: object,
        *,
        user_id: str,
        limit: int,
        status: str | None = None,
        cursor_started_at: datetime | None = None,
        cursor_session_id: str | None = None,
    ) -> list[dict]:
        sql = [f"SELECT {SESSION_COLUMNS} FROM sessions WHERE user_id = %s"]
        params: list[object] = [user_id]
        if status is not None:
            sql.append("AND status = %s")
            params.append(status)
        if cursor_started_at is not None and cursor_session_id is not None:
            sql.append("AND (started_at, session_id) < (%s, %s)")
            params.extend([cursor_started_at, cursor_session_id])
        sql.append("ORDER BY started_at DESC, session_id DESC")
        sql.append("LIMIT %s")
        params.append(limit)
        with conn.cursor() as cur:
            cur.execute("\n".join(sql), params)
            return list(cur.fetchall())
