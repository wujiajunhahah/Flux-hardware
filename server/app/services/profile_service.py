from __future__ import annotations

import json
from typing import Any

from fastapi.encoders import jsonable_encoder

from ..core.auth import AuthPrincipal
from ..core.db import connection
from ..core.errors import AppError
from ..core.ids import new_id
from ..core.security import utcnow
from ..domain.profile import UpdateDeviceCalibrationRequest, UpdateProfileRequest
from ..repositories.device_repository import DeviceRepository
from ..repositories.profile_repository import ProfileRepository


class ProfileService:
    def __init__(
        self,
        profile_repository: ProfileRepository | None = None,
        device_repository: DeviceRepository | None = None,
    ) -> None:
        self.profile_repository = profile_repository or ProfileRepository()
        self.device_repository = device_repository or DeviceRepository()

    def get_profile(self, principal: AuthPrincipal) -> dict:
        now = utcnow()
        with connection() as conn:
            try:
                profile = self._ensure_profile(conn, principal.user_id, now)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return {"profile_state": self._serialize_profile(profile)}

    def update_profile(self, principal: AuthPrincipal, payload: UpdateProfileRequest) -> dict:
        now = utcnow()
        summary_json = json.dumps(jsonable_encoder(payload.summary.model_dump(exclude_none=True)))

        with connection() as conn:
            try:
                current = self._ensure_profile(conn, principal.user_id, now, for_update=True)
                if payload.base_version != current["version"]:
                    raise AppError(
                        409,
                        "profile_version_conflict",
                        "base_version is stale",
                        data={
                            "current_profile_state": {
                                "profile_id": current["profile_id"],
                                "version": current["version"],
                                "updated_at": current["updated_at"],
                            }
                        },
                    )
                if payload.active_model_release_id and not self.profile_repository.model_release_exists(
                    conn,
                    payload.active_model_release_id,
                ):
                    raise AppError(404, "resource_not_found", "model release not found")

                updated = self.profile_repository.update_profile_state(
                    conn,
                    user_id=principal.user_id,
                    current_version=current["version"],
                    calibration_offset=payload.calibration_offset,
                    estimated_accuracy=payload.estimated_accuracy,
                    training_count=payload.training_count,
                    active_model_release_id=payload.active_model_release_id,
                    summary_json=summary_json,
                    updated_at=now,
                )
                conn.commit()
            except AppError:
                conn.rollback()
                raise
            except Exception:
                conn.rollback()
                raise

        return {"profile_state": self._serialize_profile(updated)}

    def get_device_calibration(self, principal: AuthPrincipal, device_id: str) -> dict:
        now = utcnow()
        with connection() as conn:
            try:
                device = self.device_repository.get_owned_device(
                    conn,
                    user_id=principal.user_id,
                    device_id=device_id,
                    include_revoked=True,
                )
                if device is None:
                    raise AppError(404, "device_not_found", "device not found")
                calibration = self._ensure_device_calibration(
                    conn,
                    user_id=principal.user_id,
                    device_id=device_id,
                    device_name=device["device_name"],
                    now=now,
                )
                conn.commit()
            except AppError:
                conn.rollback()
                raise
            except Exception:
                conn.rollback()
                raise

        return {"device_calibration": self._serialize_device_calibration(calibration)}

    def update_device_calibration(
        self,
        principal: AuthPrincipal,
        device_id: str,
        payload: UpdateDeviceCalibrationRequest,
    ) -> dict:
        now = utcnow()
        sensor_profile_json = json.dumps(jsonable_encoder(payload.sensor_profile))

        with connection() as conn:
            try:
                device = self.device_repository.get_owned_device(
                    conn,
                    user_id=principal.user_id,
                    device_id=device_id,
                    include_revoked=False,
                )
                if device is None:
                    raise AppError(404, "device_not_found", "device not found")

                current = self._ensure_device_calibration(
                    conn,
                    user_id=principal.user_id,
                    device_id=device_id,
                    device_name=device["device_name"],
                    now=now,
                    for_update=True,
                )
                if payload.base_version != current["version"]:
                    raise AppError(
                        409,
                        "device_calibration_version_conflict",
                        "base_version is stale",
                        data={
                            "current_device_calibration": {
                                "device_id": current["device_id"],
                                "version": current["version"],
                                "updated_at": current["updated_at"],
                            }
                        },
                    )

                updated = self.profile_repository.update_device_calibration(
                    conn,
                    user_id=principal.user_id,
                    device_id=device_id,
                    current_version=current["version"],
                    device_name=payload.device_name,
                    sensor_profile_json=sensor_profile_json,
                    calibration_offset=payload.calibration_offset,
                    updated_at=now,
                )
                conn.commit()
            except AppError:
                conn.rollback()
                raise
            except Exception:
                conn.rollback()
                raise

        return {"device_calibration": self._serialize_device_calibration(updated)}

    def _ensure_profile(
        self,
        conn: object,
        user_id: str,
        now: object,
        *,
        for_update: bool = False,
    ) -> dict:
        profile = self.profile_repository.get_profile_state(conn, user_id, for_update=for_update)
        if profile is not None:
            return profile
        created = self.profile_repository.create_profile_state(
            conn,
            profile_id=new_id("pro"),
            user_id=user_id,
            now=now,
        )
        if created is None:
            return self.profile_repository.get_profile_state(conn, user_id, for_update=for_update)
        if for_update:
            return self.profile_repository.get_profile_state(conn, user_id, for_update=True)
        return created

    def _ensure_device_calibration(
        self,
        conn: object,
        *,
        user_id: str,
        device_id: str,
        device_name: str,
        now: object,
        for_update: bool = False,
    ) -> dict:
        calibration = self.profile_repository.get_device_calibration(
            conn,
            user_id=user_id,
            device_id=device_id,
            for_update=for_update,
        )
        if calibration is not None:
            return calibration
        created = self.profile_repository.create_device_calibration(
            conn,
            user_id=user_id,
            device_id=device_id,
            device_name=device_name,
            now=now,
        )
        if created is None:
            return self.profile_repository.get_device_calibration(
                conn,
                user_id=user_id,
                device_id=device_id,
                for_update=for_update,
            )
        if for_update:
            return self.profile_repository.get_device_calibration(
                conn,
                user_id=user_id,
                device_id=device_id,
                for_update=True,
            )
        return created

    def _serialize_profile(self, row: dict) -> dict:
        return {
            "profile_id": row["profile_id"],
            "user_id": row["user_id"],
            "version": row["version"],
            "calibration_offset": float(row["calibration_offset"]),
            "estimated_accuracy": float(row["estimated_accuracy"]),
            "training_count": row["training_count"],
            "active_model_release_id": row["active_model_release_id"],
            "summary": self._coerce_json(row["summary_json"]),
            "updated_at": row["updated_at"],
        }

    def _serialize_device_calibration(self, row: dict) -> dict:
        return {
            "device_id": row["device_id"],
            "user_id": row["user_id"],
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
