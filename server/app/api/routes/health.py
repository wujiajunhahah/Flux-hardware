from fastapi import APIRouter

from ...core.settings import settings

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }
