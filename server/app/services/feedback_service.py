from __future__ import annotations

from datetime import datetime
from typing import Any

from ..core.auth import AuthPrincipal
from ..core.db import connection
from ..core.errors import AppError
from ..core.pagination import decode_cursor, encode_cursor
from ..domain.feedback import CreateFeedbackEventRequest
from ..repositories.device_repository import DeviceRepository
from ..repositories.feedback_repository import FeedbackRepository
from ..repositories.session_repository import SessionRepository
from .device_guard import require_current_device


class FeedbackService:
    def __init__(
        self,
        feedback_repository: FeedbackRepository | None = None,
        device_repository: DeviceRepository | None = None,
        session_repository: SessionRepository | None = None,
    ) -> None:
        self.feedback_repository = feedback_repository or FeedbackRepository()
        self.device_repository = device_repository or DeviceRepository()
        self.session_repository = session_repository or SessionRepository()

    def create_feedback_event(
        self,
        principal: AuthPrincipal,
        payload: CreateFeedbackEventRequest,
        idempotency_key: str,
    ) -> dict[str, Any]:
        with connection() as conn:
            try:
                require_current_device(
                    conn,
                    principal=principal,
                    device_id=payload.device_id,
                    device_repository=self.device_repository,
                )
                if payload.session_id is not None:
                    session = self.session_repository.get_session(conn, principal.user_id, payload.session_id)
                    if session is None:
                        raise AppError(404, "resource_not_found", "session not found")

                existing = self.feedback_repository.get_feedback_by_idempotency(conn, principal.user_id, idempotency_key)
                if existing is not None:
                    conn.commit()
                    return {"feedback_event": self._serialize_feedback(existing)}

                duplicate_feedback = self.feedback_repository.get_feedback_event_any(conn, payload.feedback_event_id)
                if duplicate_feedback is not None:
                    raise AppError(409, "feedback_event_already_exists", "feedback event already exists")

                row = self.feedback_repository.create_feedback_event(
                    conn,
                    feedback_event_id=payload.feedback_event_id,
                    user_id=principal.user_id,
                    device_id=payload.device_id,
                    session_id=payload.session_id,
                    idempotency_key=idempotency_key,
                    predicted_stamina=payload.predicted_stamina,
                    actual_stamina=payload.actual_stamina,
                    label=payload.label,
                    kss=payload.kss,
                    note=payload.note,
                    created_at=payload.created_at,
                )
                conn.commit()
            except AppError:
                conn.rollback()
                raise
            except Exception:
                conn.rollback()
                raise

        return {"feedback_event": self._serialize_feedback(row)}

    def list_feedback_events(
        self,
        principal: AuthPrincipal,
        *,
        cursor: str | None,
        limit: int,
        session_id: str | None,
    ) -> dict[str, Any]:
        cursor_created_at: datetime | None = None
        cursor_feedback_event_id: str | None = None
        if cursor is not None:
            try:
                decoded = decode_cursor(cursor)
                cursor_created_at = datetime.fromisoformat(decoded["created_at"])
                cursor_feedback_event_id = decoded["feedback_event_id"]
            except Exception:
                raise AppError(400, "invalid_cursor", "cursor is invalid")

        with connection() as conn:
            rows = self.feedback_repository.list_feedback_events(
                conn,
                user_id=principal.user_id,
                limit=limit,
                session_id=session_id,
                cursor_created_at=cursor_created_at,
                cursor_feedback_event_id=cursor_feedback_event_id,
            )

        next_cursor = None
        if len(rows) == limit:
            last = rows[-1]
            next_cursor = encode_cursor(
                {
                    "created_at": last["created_at"].isoformat(),
                    "feedback_event_id": last["feedback_event_id"],
                }
            )

        return {
            "items": [self._serialize_feedback(row) for row in rows],
            "next_cursor": next_cursor,
        }

    def _serialize_feedback(self, row: dict) -> dict[str, Any]:
        return {
            "feedback_event_id": row["feedback_event_id"],
            "user_id": row["user_id"],
            "device_id": row["device_id"],
            "session_id": row["session_id"],
            "predicted_stamina": row["predicted_stamina"],
            "actual_stamina": row["actual_stamina"],
            "label": row["label"],
            "kss": row["kss"],
            "note": row["note"],
            "created_at": row["created_at"],
        }
