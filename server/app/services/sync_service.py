from __future__ import annotations

import json
from typing import Any

from ..core.auth import AuthPrincipal
from ..core.db import connection
from ..core.ids import new_id
from ..core.security import utcnow
from ..repositories.device_repository import DeviceRepository
from ..repositories.profile_repository import ProfileRepository
from .device_guard import require_current_device
from .model_service import ModelService


class SyncService:
    def __init__(
        self,
        profile_repository: ProfileRepository | None = None,
        device_repository: DeviceRepository | None = None,
        model_service: ModelService | None = None,
    ) -> None:
        self.profile_repository = profile_repository or ProfileRepository()
        self.device_repository = device_repository or DeviceRepository()
        self.model_service = model_service or ModelService()

    def get_bootstrap(
        self,
        principal: AuthPrincipal,
        *,
        device_id: str,
        platform: str,
        channel: str,
        build_model_url: Any,
    ) -> dict[str, Any]:
        now = utcnow()
        with connection() as conn:
            require_current_device(
                conn,
                principal=principal,
                device_id=device_id,
                device_repository=self.device_repository,
            )
            profile = self.profile_repository.get_profile_state(conn, principal.user_id)
            if profile is None:
                profile = self.profile_repository.create_profile_state(
                    conn,
                    profile_id=new_id("pro"),
                    user_id=principal.user_id,
                    now=now,
                )
                if profile is None:
                    profile = self.profile_repository.get_profile_state(conn, principal.user_id)
            calibrations = self.profile_repository.list_device_calibrations(conn, user_id=principal.user_id)
            model_row = self.model_service.resolve_release_for_profile(
                conn=conn,
                active_model_release_id=profile["active_model_release_id"] if profile is not None else None,
                platform=platform,
                channel=channel,
            )
            conn.commit()

        manifest = self.model_service.serialize_manifest(model_row)
        if manifest is not None:
            manifest["artifact_url"] = self.model_service.build_artifact_url(model_row, build_model_url)

        return {
            "server_time": now,
            "profile_state": self._serialize_profile(profile),
            "device_calibrations": [self._serialize_device_calibration(row) for row in calibrations],
            "active_model_manifest": manifest,
        }

    def _serialize_profile(self, row: dict | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "profile_id": row["profile_id"],
            "version": row["version"],
            "calibration_offset": float(row["calibration_offset"]),
            "estimated_accuracy": float(row["estimated_accuracy"]),
            "training_count": row["training_count"],
            "active_model_release_id": row["active_model_release_id"],
            "summary": self._coerce_json(row["summary_json"]),
            "updated_at": row["updated_at"],
        }

    def _serialize_device_calibration(self, row: dict) -> dict[str, Any]:
        return {
            "device_id": row["device_id"],
            "version": row["version"],
            "device_name": row["device_name"],
            "sensor_profile": self._coerce_json(row["sensor_profile_json"]),
            "calibration_offset": float(row["calibration_offset"]),
            "updated_at": row["updated_at"],
        }

    def _coerce_json(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, str):
            return json.loads(value)
        return value
