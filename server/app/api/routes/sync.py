from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Request

from ...core.auth import AuthPrincipal, require_principal
from ...core.errors import ok
from ...services.sync_service import SyncService

router = APIRouter()
service = SyncService()


@router.get("/sync/bootstrap")
def get_sync_bootstrap(
    request: Request,
    device_id: str,
    platform: Literal["ios", "android", "web"],
    channel: Literal["stable", "beta", "internal"] = "stable",
    principal: AuthPrincipal = Depends(require_principal),
):
    data = service.get_bootstrap(
        principal,
        device_id=device_id,
        platform=platform,
        channel=channel,
        build_model_url=lambda route_name, **params: request.url_for(route_name, release_id=params["release_id"]).include_query_params(token=params["token"]),
    )
    return ok(request, data)
