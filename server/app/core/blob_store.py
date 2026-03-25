from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
from pathlib import Path, PurePosixPath

from .security import import_jwt, utcnow
from .settings import REPO_ROOT, settings

SESSION_BLOB_AUDIENCE = "fluxchi-platform-session-blob"
SESSION_BLOB_ROOT = REPO_ROOT / "data" / "platform-session-blobs"


@dataclass(frozen=True)
class SessionBlobTokenClaims:
    session_id: str
    object_key: str
    operation: str
    content_type: str
    issued_at: datetime
    expires_at: datetime



def build_session_object_key(session_id: str, started_at: datetime, content_type: str) -> str:
    started_at_utc = started_at.astimezone(UTC)
    extension = ".json" if content_type == "application/json" else ".bin"
    return (
        f"sessions/{started_at_utc:%Y/%m/%d}/{session_id}{extension}"
    )



def build_session_blob_token(
    *,
    session_id: str,
    object_key: str,
    operation: str,
    content_type: str,
    expires_at: datetime | None = None,
) -> str:
    jwt = import_jwt()
    now = utcnow()
    expires_at = expires_at or now + timedelta(seconds=settings.upload_url_ttl_sec)
    payload = {
        "sub": session_id,
        "object_key": normalize_object_key(object_key),
        "type": f"session_blob:{operation}",
        "content_type": content_type,
        "iss": settings.access_token_issuer,
        "aud": SESSION_BLOB_AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.access_token_algorithm)



def decode_session_blob_token(token: str, *, expected_operation: str) -> SessionBlobTokenClaims:
    jwt = import_jwt()
    payload = jwt.decode(
        token,
        settings.secret_key,
        algorithms=[settings.access_token_algorithm],
        audience=SESSION_BLOB_AUDIENCE,
        issuer=settings.access_token_issuer,
    )
    expected_type = f"session_blob:{expected_operation}"
    if payload.get("type") != expected_type:
        raise jwt.InvalidTokenError("unexpected token type")
    return SessionBlobTokenClaims(
        session_id=payload["sub"],
        object_key=normalize_object_key(payload["object_key"]),
        operation=expected_operation,
        content_type=payload["content_type"],
        issued_at=datetime.fromtimestamp(payload["iat"], tz=UTC),
        expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC),
    )



def normalize_object_key(object_key: str) -> str:
    path = PurePosixPath(object_key)
    normalized = str(path)
    if not normalized or normalized == "." or path.is_absolute() or ".." in path.parts:
        raise ValueError("invalid object key")
    return normalized



def ensure_blob_root() -> None:
    SESSION_BLOB_ROOT.mkdir(parents=True, exist_ok=True)



def path_for_object_key(object_key: str) -> Path:
    ensure_blob_root()
    normalized = normalize_object_key(object_key)
    return SESSION_BLOB_ROOT.joinpath(*PurePosixPath(normalized).parts)



def put_blob(object_key: str, payload: bytes) -> Path:
    path = path_for_object_key(object_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("wb") as handle:
        handle.write(payload)
    temp_path.replace(path)
    return path



def blob_metadata(object_key: str) -> dict[str, object] | None:
    path = path_for_object_key(object_key)
    if not path.is_file():
        return None

    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            digest.update(chunk)

    return {
        "path": path,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }



def filename_for_object_key(object_key: str) -> str:
    return PurePosixPath(normalize_object_key(object_key)).name
