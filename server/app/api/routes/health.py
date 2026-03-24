from fastapi import APIRouter, Request

from ...core.errors import ok
from ...core.settings import settings

router = APIRouter()


@router.get("/health")
def health(request: Request):
    return ok(
        request,
        {
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
        },
    )
