from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


def new_request_id() -> str:
    return f"req_{uuid4().hex[:26]}"


def request_id_for(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if request_id is None:
        request_id = new_request_id()
        request.state.request_id = request_id
    return request_id


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> JSONResponse:
        request.state.request_id = request.headers.get("X-Request-ID") or new_request_id()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response


@dataclass
class AppError(Exception):
    status_code: int
    code: str
    message: str
    data: dict[str, Any] | None = None


def ok(request: Request, data: dict[str, Any] | None = None, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(
            {
                "ok": True,
                "request_id": request_id_for(request),
                "data": data or {},
            }
        ),
    )


def install_request_context(app: FastAPI) -> None:
    app.add_middleware(RequestContextMiddleware)


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        payload: dict[str, Any] = {
            "ok": False,
            "request_id": request_id_for(request),
            "error": {
                "code": exc.code,
                "message": exc.message,
            },
        }
        if exc.data is not None:
            payload["data"] = exc.data
        return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(payload))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                {
                    "ok": False,
                    "request_id": request_id_for(request),
                    "error": {
                        "code": "validation_error",
                        "message": "request validation failed",
                    },
                    "data": {
                        "details": exc.errors(),
                    },
                }
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder(
                {
                    "ok": False,
                    "request_id": request_id_for(request),
                    "error": {
                        "code": "internal_error",
                        "message": "internal server error",
                    },
                }
            ),
        )
