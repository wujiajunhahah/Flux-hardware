from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

from .blob_store import normalize_object_key
from .security import import_jwt, utcnow
from .settings import REPO_ROOT, settings

MODEL_ARTIFACT_AUDIENCE = "fluxchi-platform-model-artifact"
MODEL_ARTIFACT_ROOT = REPO_ROOT / "data" / "platform-model-artifacts"


@dataclass(frozen=True)
class ModelArtifactTokenClaims:
    model_release_id: str
    object_key: str
    issued_at: datetime
    expires_at: datetime


def build_model_artifact_token(
    *,
    model_release_id: str,
    object_key: str,
    expires_at: datetime | None = None,
) -> str:
    jwt = import_jwt()
    now = utcnow()
    expires_at = expires_at or now + timedelta(seconds=settings.upload_url_ttl_sec)
    payload = {
        "sub": model_release_id,
        "object_key": normalize_object_key(object_key),
        "type": "model_artifact:download",
        "iss": settings.access_token_issuer,
        "aud": MODEL_ARTIFACT_AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.access_token_algorithm)


def decode_model_artifact_token(token: str) -> ModelArtifactTokenClaims:
    jwt = import_jwt()
    payload = jwt.decode(
        token,
        settings.secret_key,
        algorithms=[settings.access_token_algorithm],
        audience=MODEL_ARTIFACT_AUDIENCE,
        issuer=settings.access_token_issuer,
    )
    if payload.get("type") != "model_artifact:download":
        raise jwt.InvalidTokenError("unexpected token type")
    return ModelArtifactTokenClaims(
        model_release_id=payload["sub"],
        object_key=normalize_object_key(payload["object_key"]),
        issued_at=datetime.fromtimestamp(payload["iat"], tz=UTC),
        expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC),
    )


def model_artifact_path(object_key: str) -> Path:
    MODEL_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    normalized = normalize_object_key(object_key)
    return MODEL_ARTIFACT_ROOT.joinpath(*PurePosixPath(normalized).parts)
