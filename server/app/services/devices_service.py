from __future__ import annotations

from ..core.auth import AuthPrincipal
from ..core.db import connection
from ..core.errors import AppError
from ..core.security import utcnow
from ..repositories.auth_repository import AuthRepository
from ..repositories.device_repository import DeviceRepository


class DevicesService:
    def __init__(
        self,
        device_repository: DeviceRepository | None = None,
        auth_repository: AuthRepository | None = None,
    ) -> None:
        self.device_repository = device_repository or DeviceRepository()
        self.auth_repository = auth_repository or AuthRepository()

    def list_devices(self, principal: AuthPrincipal) -> dict:
        with connection() as conn:
            rows = self.device_repository.list_active_devices(conn, principal.user_id)

        return {
            "items": [
                {
                    "device_id": row["device_id"],
                    "platform": row["platform"],
                    "device_name": row["device_name"],
                    "app_version": row["app_version"],
                    "last_seen_at": row["last_seen_at"],
                    "is_current_device": row["device_id"] == principal.device_id,
                }
                for row in rows
            ]
        }

    def delete_device(self, principal: AuthPrincipal, device_id: str) -> dict:
        now = utcnow()
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

                revoked = self.device_repository.revoke_device(
                    conn,
                    user_id=principal.user_id,
                    device_id=device_id,
                    revoked_at=now,
                )
                self.auth_repository.revoke_refresh_tokens_for_device(
                    conn,
                    user_id=principal.user_id,
                    device_id=device_id,
                    revoked_at=now,
                )
                conn.commit()
            except AppError:
                conn.rollback()
                raise
            except Exception:
                conn.rollback()
                raise

        return {
            "device": {
                "device_id": revoked["device_id"],
                "revoked_at": revoked["revoked_at"],
            }
        }
