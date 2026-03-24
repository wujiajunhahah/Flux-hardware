from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import secrets
from typing import Any

from .settings import settings


class SecurityDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class AccessTokenClaims:
    user_id: str
    device_id: str
    issued_at: datetime
    expires_at: datetime


def import_jwt() -> Any:
    try:
        import jwt
    except ImportError as exc:  # pragma: no cover - dependency may be absent during bootstrap
        raise SecurityDependencyError(
            "PyJWT is required for token operations. Install requirements before enabling auth routes."
        ) from exc
    return jwt


def utcnow() -> datetime:
    return datetime.now(UTC)


def build_access_token(
    *,
    user_id: str,
    device_id: str,
    issued_at: datetime | None = None,
    ttl_sec: int | None = None,
) -> str:
    jwt = import_jwt()
    now = issued_at or utcnow()
    expires_at = now + timedelta(seconds=ttl_sec or settings.access_token_ttl_sec)
    payload = {
        "sub": user_id,
        "device_id": device_id,
        "type": "access",
        "iss": settings.access_token_issuer,
        "aud": settings.access_token_audience,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    return jwt.encode(
        payload,
        settings.secret_key,
        algorithm=settings.access_token_algorithm,
    )


def decode_access_token(token: str) -> AccessTokenClaims:
    jwt = import_jwt()
    payload = jwt.decode(
        token,
        settings.secret_key,
        algorithms=[settings.access_token_algorithm],
        audience=settings.access_token_audience,
        issuer=settings.access_token_issuer,
    )
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("unexpected token type")
    return AccessTokenClaims(
        user_id=payload["sub"],
        device_id=payload["device_id"],
        issued_at=datetime.fromtimestamp(payload["iat"], tz=UTC),
        expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC),
    )


def generate_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_token(token: str) -> str:
    # Only use this for storing high-entropy random tokens, not user passwords.
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
