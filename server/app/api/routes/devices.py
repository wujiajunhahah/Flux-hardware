from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ...core.auth import AuthPrincipal, require_principal
from ...core.errors import ok
from ...services.devices_service import DevicesService

router = APIRouter()
service = DevicesService()


@router.get("/devices")
def list_devices(request: Request, principal: AuthPrincipal = Depends(require_principal)):
    return ok(request, service.list_devices(principal))


@router.delete("/devices/{device_id}")
def delete_device(
    device_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(require_principal),
):
    return ok(request, service.delete_device(principal, device_id))
