from __future__ import annotations

from datetime import datetime

FEEDBACK_COLUMNS = """
    feedback_event_id,
    user_id,
    device_id,
    session_id,
    idempotency_key,
    predicted_stamina,
    actual_stamina,
    label,
    kss,
    note,
    created_at
"""


class FeedbackRepository:
    def get_feedback_event(self, conn: object, user_id: str, feedback_event_id: str) -> dict | None:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {FEEDBACK_COLUMNS} FROM feedback_events WHERE user_id = %s AND feedback_event_id = %s",
                (user_id, feedback_event_id),
            )
            return cur.fetchone()

    def get_feedback_event_any(self, conn: object, feedback_event_id: str) -> dict | None:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {FEEDBACK_COLUMNS} FROM feedback_events WHERE feedback_event_id = %s",
                (feedback_event_id,),
            )
            return cur.fetchone()

    def get_feedback_by_idempotency(self, conn: object, user_id: str, idempotency_key: str) -> dict | None:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {FEEDBACK_COLUMNS} FROM feedback_events WHERE user_id = %s AND idempotency_key = %s",
                (user_id, idempotency_key),
            )
            return cur.fetchone()

    def create_feedback_event(
        self,
        conn: object,
        *,
        feedback_event_id: str,
        user_id: str,
        device_id: str,
        session_id: str | None,
        idempotency_key: str,
        predicted_stamina: int,
        actual_stamina: int,
        label: str,
        kss: int | None,
        note: str | None,
        created_at: datetime,
    ) -> dict:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO feedback_events (
                    feedback_event_id,
                    user_id,
                    device_id,
                    session_id,
                    idempotency_key,
                    predicted_stamina,
                    actual_stamina,
                    label,
                    kss,
                    note,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {FEEDBACK_COLUMNS}
                """,
                (
                    feedback_event_id,
                    user_id,
                    device_id,
                    session_id,
                    idempotency_key,
                    predicted_stamina,
                    actual_stamina,
                    label,
                    kss,
                    note,
                    created_at,
                ),
            )
            return cur.fetchone()

    def list_feedback_events(
        self,
        conn: object,
        *,
        user_id: str,
        limit: int,
        session_id: str | None = None,
        cursor_created_at: datetime | None = None,
        cursor_feedback_event_id: str | None = None,
    ) -> list[dict]:
        sql = [f"SELECT {FEEDBACK_COLUMNS} FROM feedback_events WHERE user_id = %s"]
        params: list[object] = [user_id]
        if session_id is not None:
            sql.append("AND session_id = %s")
            params.append(session_id)
        if cursor_created_at is not None and cursor_feedback_event_id is not None:
            sql.append("AND (created_at, feedback_event_id) < (%s, %s)")
            params.extend([cursor_created_at, cursor_feedback_event_id])
        sql.append("ORDER BY created_at DESC, feedback_event_id DESC")
        sql.append("LIMIT %s")
        params.append(limit)
        with conn.cursor() as cur:
            cur.execute("\n".join(sql), params)
            return list(cur.fetchall())
