from fastapi import FastAPI

from .api.router import api_router
from .core.settings import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
