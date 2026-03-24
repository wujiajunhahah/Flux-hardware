from __future__ import annotations

from fastapi import APIRouter, Request

from ...core.errors import ok
from ...domain.auth import RefreshRequest, SignInRequest, SignOutRequest
from ...services.auth_service import AuthService

router = APIRouter(prefix="/auth")
service = AuthService()


@router.post("/sign-up")
def sign_up(request: Request, payload: SignInRequest):
    return ok(request, service.sign_up(payload))


@router.post("/sign-in")
def sign_in(request: Request, payload: SignInRequest):
    return ok(request, service.sign_in(payload))


@router.post("/refresh")
def refresh_access_token(request: Request, payload: RefreshRequest):
    return ok(request, service.refresh_access_token(payload))


@router.post("/sign-out")
def sign_out(request: Request, payload: SignOutRequest):
    return ok(request, service.sign_out(payload))
