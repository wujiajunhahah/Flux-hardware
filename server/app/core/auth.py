from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..repositories.auth_repository import AuthRepository
from .db import connection
from .errors import AppError
from .security import decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthPrincipal:
    user_id: str
    device_id: str


_auth_repository = AuthRepository()


def require_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthPrincipal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AppError(401, "unauthorized", "bearer token is required")

    try:
        claims = decode_access_token(credentials.credentials)
    except Exception:
        raise AppError(401, "unauthorized", "token is invalid")

    with connection() as conn:
        principal = _auth_repository.get_active_principal(conn, claims.user_id, claims.device_id)

    if principal is None:
        raise AppError(401, "unauthorized", "token is invalid")

    return AuthPrincipal(user_id=claims.user_id, device_id=claims.device_id)
