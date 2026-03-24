from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ...core.auth import AuthPrincipal, require_principal
from ...core.errors import ok
from ...domain.profile import UpdateDeviceCalibrationRequest, UpdateProfileRequest
from ...services.profile_service import ProfileService

router = APIRouter()
service = ProfileService()


@router.get("/profile")
def get_profile(request: Request, principal: AuthPrincipal = Depends(require_principal)):
    return ok(request, service.get_profile(principal))


@router.put("/profile")
def update_profile(
    request: Request,
    payload: UpdateProfileRequest,
    principal: AuthPrincipal = Depends(require_principal),
):
    return ok(request, service.update_profile(principal, payload))


@router.get("/devices/{device_id}/calibration")
def get_device_calibration(
    device_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(require_principal),
):
    return ok(request, service.get_device_calibration(principal, device_id))


@router.put("/devices/{device_id}/calibration")
def update_device_calibration(
    device_id: str,
    request: Request,
    payload: UpdateDeviceCalibrationRequest,
    principal: AuthPrincipal = Depends(require_principal),
):
    return ok(request, service.update_device_calibration(principal, device_id, payload))
