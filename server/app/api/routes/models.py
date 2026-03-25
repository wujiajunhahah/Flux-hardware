from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse

from ...core.auth import AuthPrincipal, require_principal
from ...core.errors import AppError, ok
from ...services.model_service import ModelService

router = APIRouter()
service = ModelService()


@router.get("/models/manifest")
def get_model_manifest(
    request: Request,
    platform: Literal["ios", "android", "web"],
    channel: Literal["stable", "beta", "internal"] = "stable",
    device_class: str | None = None,
    principal: AuthPrincipal = Depends(require_principal),
):
    del device_class
    data = service.get_active_manifest(
        principal,
        platform=platform,
        channel=channel,
        build_model_url=lambda route_name, **params: request.url_for(route_name, release_id=params["release_id"]).include_query_params(token=params["token"]),
    )
    return ok(request, data)


@router.get("/model-artifacts/{release_id}", include_in_schema=False, name="get_model_artifact")
def get_model_artifact(release_id: str, token: Annotated[str, Query(min_length=1)]):
    try:
        artifact = service.download_model_artifact(release_id=release_id, token=token)
    except FileNotFoundError:
        raise AppError(404, "resource_not_found", "model artifact not found")
    except ValueError:
        raise AppError(401, "unauthorized", "artifact token is invalid")
    return FileResponse(path=str(artifact["path"]), filename=artifact["filename"])
