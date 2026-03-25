from __future__ import annotations

from ..core.auth import AuthPrincipal
from ..core.errors import AppError
from ..repositories.device_repository import DeviceRepository


def require_current_device(
    conn: object,
    *,
    principal: AuthPrincipal,
    device_id: str,
    device_repository: DeviceRepository,
) -> None:
    if device_id != principal.device_id:
        raise AppError(403, "forbidden", "device_id does not match authenticated device")
    device = device_repository.get_owned_device(
        conn,
        user_id=principal.user_id,
        device_id=device_id,
        include_revoked=False,
    )
    if device is None:
        raise AppError(404, "device_not_found", "device not found")
