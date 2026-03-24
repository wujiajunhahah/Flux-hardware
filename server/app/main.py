from fastapi import FastAPI

from .api.router import api_router
from .core.errors import install_exception_handlers, install_request_context
from .core.settings import settings


def create_app() -> FastAPI:
    if settings.is_production and settings.secret_key_is_default:
        raise RuntimeError("FLUX_SECRET_KEY must be set in production")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    install_request_context(app)
    install_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
