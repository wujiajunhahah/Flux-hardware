from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from ..core.auth import AuthPrincipal
from ..core.db import connection
from ..core.model_store import build_model_artifact_token, decode_model_artifact_token, model_artifact_path
from ..core.security import utcnow
from ..core.settings import settings
from ..repositories.model_repository import ModelRepository


class ModelService:
    def __init__(self, model_repository: ModelRepository | None = None) -> None:
        self.model_repository = model_repository or ModelRepository()

    def get_active_manifest(
        self,
        principal: AuthPrincipal,
        *,
        platform: str,
        channel: str,
        build_model_url: Any,
    ) -> dict[str, Any]:
        del principal  # V1 keeps this endpoint authenticated but user-agnostic.
        with connection() as conn:
            row = self.model_repository.get_latest_published_model_release(
                conn,
                platform=platform,
                channel=channel,
            )
        manifest = self.serialize_manifest(row)
        if manifest is not None:
            manifest["artifact_url"] = self.build_artifact_url(row, build_model_url)
        return {"active_model_manifest": manifest}

    def build_artifact_url(self, row: dict | None, build_url: Any) -> str | None:
        if row is None:
            return None
        token = build_model_artifact_token(
            model_release_id=row["model_release_id"],
            object_key=row["artifact_object_key"],
            expires_at=utcnow() + timedelta(seconds=settings.upload_url_ttl_sec),
        )
        return str(build_url("get_model_artifact", release_id=row["model_release_id"], token=token))

    def download_model_artifact(self, *, release_id: str, token: str) -> dict[str, Any]:
        try:
            claims = decode_model_artifact_token(token)
        except Exception:
            raise ValueError("artifact token is invalid")
        if claims.model_release_id != release_id:
            raise ValueError("artifact token is invalid")

        with connection() as conn:
            row = self.model_repository.get_model_release(conn, release_id)
        if row is None or row["status"] != "published" or row["artifact_object_key"] != claims.object_key:
            raise ValueError("artifact token is invalid")

        path = model_artifact_path(claims.object_key)
        if not path.is_file():
            raise FileNotFoundError(claims.object_key)

        return {
            "path": path,
            "filename": path.name,
        }

    def resolve_release_for_profile(
        self,
        *,
        conn: object,
        active_model_release_id: str | None,
        platform: str,
        channel: str,
    ) -> dict | None:
        if active_model_release_id:
            row = self.model_repository.get_model_release(conn, active_model_release_id)
            if (
                row is not None
                and row["status"] == "published"
                and row["platform"] == platform
                and row["channel"] == channel
                and row["model_kind"] == "global_base_model"
            ):
                return row
        return self.model_repository.get_latest_published_model_release(
            conn,
            platform=platform,
            channel=channel,
        )

    def serialize_manifest(self, row: dict | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "model_release_id": row["model_release_id"],
            "platform": row["platform"],
            "channel": row["channel"],
            "model_kind": row["model_kind"],
            "version": row["version"],
            "artifact_sha256": row["artifact_sha256"],
            "config": self._coerce_json(row["config_json"]),
            "published_at": row["published_at"],
        }

    def _coerce_json(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, str):
            return json.loads(value)
        return value
