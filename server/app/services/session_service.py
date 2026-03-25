from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..core.auth import AuthPrincipal
from ..core.blob_store import (
    build_session_blob_token,
    build_session_object_key,
    blob_metadata,
    decode_session_blob_token,
    filename_for_object_key,
    put_blob,
)
from ..core.db import connection
from ..core.errors import AppError
from ..core.pagination import decode_cursor, encode_cursor
from ..core.security import utcnow
from ..core.settings import settings
from ..domain.session import CreateSessionRequest, FinalizeSessionRequest
from ..repositories.device_repository import DeviceRepository
from ..repositories.session_repository import SessionRepository
from .device_guard import require_current_device


class SessionService:
    def __init__(
        self,
        session_repository: SessionRepository | None = None,
        device_repository: DeviceRepository | None = None,
    ) -> None:
        self.session_repository = session_repository or SessionRepository()
        self.device_repository = device_repository or DeviceRepository()

    def create_session(
        self,
        principal: AuthPrincipal,
        payload: CreateSessionRequest,
        idempotency_key: str,
    ) -> dict[str, Any]:
        now = utcnow()
        self._validate_session_timestamps(payload.started_at, payload.ended_at)

        with connection() as conn:
            try:
                require_current_device(
                    conn,
                    principal=principal,
                    device_id=payload.device_id,
                    device_repository=self.device_repository,
                )
                existing = self.session_repository.get_session_by_idempotency(conn, principal.user_id, idempotency_key)
                if existing is not None:
                    conn.commit()
                    return self._serialize_create_result(existing, now)

                duplicate_session = self.session_repository.get_session_any(conn, payload.session_id)
                if duplicate_session is not None:
                    raise AppError(409, "session_already_exists", "session already exists")

                row = self.session_repository.create_session(
                    conn,
                    session_id=payload.session_id,
                    user_id=principal.user_id,
                    device_id=payload.device_id,
                    idempotency_key=idempotency_key,
                    status="pending_upload",
                    source=payload.source,
                    title=payload.title,
                    started_at=payload.started_at,
                    ended_at=payload.ended_at,
                    duration_sec=payload.duration_sec,
                    snapshot_count=payload.snapshot_count,
                    schema_version=payload.schema_version,
                    content_type=payload.content_type,
                    blob_object_key=build_session_object_key(
                        payload.session_id,
                        payload.started_at,
                        payload.content_type,
                    ),
                    blob_size_bytes=payload.size_bytes,
                    blob_sha256=payload.sha256,
                    now=now,
                )
                conn.commit()
            except AppError:
                conn.rollback()
                raise
            except Exception:
                conn.rollback()
                raise

        return self._serialize_create_result(row, now)

    def upload_session_blob(
        self,
        *,
        session_id: str,
        token: str,
        payload: bytes,
        content_type: str | None,
    ) -> None:
        try:
            claims = decode_session_blob_token(token, expected_operation="upload")
        except Exception:
            raise AppError(401, "unauthorized", "upload token is invalid")
        if claims.session_id != session_id:
            raise AppError(401, "unauthorized", "upload token is invalid")
        if content_type is not None:
            normalized = content_type.split(";", 1)[0].strip().lower()
            if normalized and normalized != claims.content_type.lower():
                raise AppError(409, "content_type_mismatch", "content type does not match session")

        with connection() as conn:
            row = self.session_repository.get_session_any(conn, session_id, for_update=True)
            if row is None or row["blob_object_key"] != claims.object_key:
                conn.rollback()
                raise AppError(401, "unauthorized", "upload token is invalid")
            if row["status"] != "pending_upload":
                conn.rollback()
                raise AppError(409, "invalid_session_status", "session upload is closed")
            conn.rollback()

        put_blob(claims.object_key, payload)

    def finalize_session(
        self,
        principal: AuthPrincipal,
        session_id: str,
        payload: FinalizeSessionRequest,
    ) -> dict[str, Any]:
        now = utcnow()

        with connection() as conn:
            try:
                row = self.session_repository.get_session(conn, principal.user_id, session_id, for_update=True)
                if row is None:
                    raise AppError(404, "resource_not_found", "session not found")
                if row["device_id"] != principal.device_id:
                    raise AppError(403, "forbidden", "session does not belong to current device")

                metadata = blob_metadata(payload.blob.object_key)
                if metadata is None:
                    raise AppError(409, "session_not_uploaded", "session blob has not been uploaded")

                expected_object_key = row["blob_object_key"]
                expected_size = int(row["blob_size_bytes"])
                expected_sha256 = row["blob_sha256"]
                actual_size = int(metadata["size_bytes"])
                actual_sha256 = str(metadata["sha256"])

                if row["status"] == "ready":
                    if (
                        payload.blob.object_key != expected_object_key
                        or payload.blob.size_bytes != expected_size
                        or payload.blob.sha256 != expected_sha256
                    ):
                        raise AppError(
                            409,
                            "session_checksum_mismatch",
                            "session metadata does not match the finalized blob",
                        )
                    conn.commit()
                    return {
                        "session": {
                            "session_id": row["session_id"],
                            "status": row["status"],
                            "download_token": self._build_download_token(row, now),
                        }
                    }

                if row["status"] != "pending_upload":
                    raise AppError(409, "invalid_session_status", "session cannot be finalized")

                if payload.blob.object_key != expected_object_key:
                    raise AppError(409, "session_checksum_mismatch", "blob object key does not match session")
                if payload.blob.size_bytes != expected_size or actual_size != expected_size:
                    raise AppError(409, "session_checksum_mismatch", "blob size does not match session")
                if payload.blob.sha256 != expected_sha256 or actual_sha256 != expected_sha256:
                    raise AppError(409, "session_checksum_mismatch", "blob sha256 does not match session")

                updated = self.session_repository.update_session_ready(
                    conn,
                    user_id=principal.user_id,
                    session_id=session_id,
                    blob_object_key=payload.blob.object_key,
                    blob_size_bytes=payload.blob.size_bytes,
                    blob_sha256=payload.blob.sha256,
                    updated_at=now,
                )
                if updated is None:
                    raise AppError(409, "session_finalize_conflict", "session could not be finalized")
                conn.commit()
            except AppError:
                conn.rollback()
                raise
            except Exception:
                conn.rollback()
                raise

        return {
            "session": {
                "session_id": updated["session_id"],
                "status": updated["status"],
                "download_token": self._build_download_token(updated, now),
            }
        }

    def list_sessions(
        self,
        principal: AuthPrincipal,
        *,
        cursor: str | None,
        limit: int,
        status: str | None,
    ) -> dict[str, Any]:
        cursor_started_at: datetime | None = None
        cursor_session_id: str | None = None
        if cursor is not None:
            try:
                decoded = decode_cursor(cursor)
                cursor_started_at = datetime.fromisoformat(decoded["started_at"])
                cursor_session_id = decoded["session_id"]
            except Exception:
                raise AppError(400, "invalid_cursor", "cursor is invalid")

        with connection() as conn:
            rows = self.session_repository.list_sessions(
                conn,
                user_id=principal.user_id,
                limit=limit,
                status=status,
                cursor_started_at=cursor_started_at,
                cursor_session_id=cursor_session_id,
            )

        next_cursor = None
        if len(rows) == limit:
            last = rows[-1]
            next_cursor = encode_cursor(
                {
                    "started_at": last["started_at"].isoformat(),
                    "session_id": last["session_id"],
                }
            )

        return {
            "items": [self._serialize_session(row) for row in rows],
            "next_cursor": next_cursor,
        }

    def get_session(self, principal: AuthPrincipal, session_id: str) -> dict[str, Any]:
        with connection() as conn:
            row = self.session_repository.get_session(conn, principal.user_id, session_id)
        if row is None:
            raise AppError(404, "resource_not_found", "session not found")
        return {"session": self._serialize_session(row)}

    def get_download_url(self, principal: AuthPrincipal, session_id: str) -> dict[str, Any]:
        now = utcnow()
        with connection() as conn:
            row = self.session_repository.get_session(conn, principal.user_id, session_id)
        if row is None:
            raise AppError(404, "resource_not_found", "session not found")
        if row["status"] != "ready":
            raise AppError(409, "session_not_uploaded", "session blob is not ready")
        metadata = blob_metadata(row["blob_object_key"])
        if metadata is None:
            raise AppError(409, "session_not_uploaded", "session blob is not ready")
        expires_at = now + timedelta(seconds=settings.upload_url_ttl_sec)
        return {
            "download_token": build_session_blob_token(
                session_id=row["session_id"],
                object_key=row["blob_object_key"],
                operation="download",
                content_type=row["content_type"],
                expires_at=expires_at,
            ),
            "expires_at": expires_at,
        }

    def download_session_blob(self, *, session_id: str, token: str) -> dict[str, Any]:
        try:
            claims = decode_session_blob_token(token, expected_operation="download")
        except Exception:
            raise AppError(401, "unauthorized", "download token is invalid")
        if claims.session_id != session_id:
            raise AppError(401, "unauthorized", "download token is invalid")

        with connection() as conn:
            row = self.session_repository.get_session_any(conn, session_id)
        if row is None or row["status"] != "ready" or row["blob_object_key"] != claims.object_key:
            raise AppError(401, "unauthorized", "download token is invalid")

        metadata = blob_metadata(claims.object_key)
        if metadata is None:
            raise AppError(409, "session_not_uploaded", "session blob is not ready")

        return {
            "path": metadata["path"],
            "content_type": row["content_type"],
            "filename": filename_for_object_key(claims.object_key),
        }

    def _serialize_create_result(self, row: dict, now: datetime) -> dict[str, Any]:
        result: dict[str, Any] = {
            "session": {
                "session_id": row["session_id"],
                "status": row["status"],
            }
        }
        if row["status"] == "pending_upload":
            expires_at = now + timedelta(seconds=settings.upload_url_ttl_sec)
            result["upload"] = {
                "object_key": row["blob_object_key"],
                "upload_token": build_session_blob_token(
                    session_id=row["session_id"],
                    object_key=row["blob_object_key"],
                    operation="upload",
                    content_type=row["content_type"],
                    expires_at=expires_at,
                ),
                "upload_method": "PUT",
                "content_type": row["content_type"],
                "expires_at": expires_at,
            }
        elif row["status"] == "ready":
            result["session"]["download_token"] = self._build_download_token(row, now)
        return result

    def _serialize_session(self, row: dict) -> dict[str, Any]:
        return {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "device_id": row["device_id"],
            "status": row["status"],
            "source": row["source"],
            "title": row["title"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "duration_sec": row["duration_sec"],
            "snapshot_count": row["snapshot_count"],
            "schema_version": row["schema_version"],
            "content_type": row["content_type"],
            "blob": {
                "object_key": row["blob_object_key"],
                "size_bytes": int(row["blob_size_bytes"]) if row["blob_size_bytes"] is not None else None,
                "sha256": row["blob_sha256"],
            },
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _build_download_token(self, row: dict, now: datetime) -> str:
        return build_session_blob_token(
            session_id=row["session_id"],
            object_key=row["blob_object_key"],
            operation="download",
            content_type=row["content_type"],
            expires_at=now + timedelta(seconds=settings.upload_url_ttl_sec),
        )

    def _validate_session_timestamps(self, started_at: datetime, ended_at: datetime) -> None:
        if ended_at < started_at:
            raise AppError(422, "validation_error", "ended_at must be greater than or equal to started_at")
