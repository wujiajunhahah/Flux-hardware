from __future__ import annotations

from dataclasses import dataclass

from .errors import AppError
from .security import import_jwt
from .settings import settings


@dataclass(frozen=True)
class ResolvedIdentity:
    provider: str
    provider_subject: str
    primary_email: str | None = None


def resolve_provider_identity(provider: str, provider_token: str) -> ResolvedIdentity:
    normalized_provider = provider.strip().lower()
    token = provider_token.strip()

    if not normalized_provider or not token:
        raise AppError(422, "validation_error", "provider and provider_token are required")

    if token.count(".") == 2:
        jwt = import_jwt()
        payload = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_aud": False,
                "verify_exp": False,
            },
        )
        subject = payload.get("sub")
        if subject:
            return ResolvedIdentity(
                provider=normalized_provider,
                provider_subject=str(subject),
                primary_email=payload.get("email"),
            )

    if settings.insecure_provider_tokens_enabled:
        subject = token[4:] if token.startswith("dev:") else token
        return ResolvedIdentity(
            provider=normalized_provider,
            provider_subject=subject,
            primary_email=None,
        )

    raise AppError(
        422,
        "validation_error",
        "provider_token cannot be resolved without a configured verifier",
    )
