from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field

SERVER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILES = (REPO_ROOT / ".env", SERVER_ROOT / ".env")

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency during early bootstrap
    load_dotenv = None

if load_dotenv is not None:
    for env_file in ENV_FILES:
        load_dotenv(env_file, override=False)

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - keep the skeleton importable before deps are installed
    BaseSettings = None
    SettingsConfigDict = None


def _env(name: str, default: Any = None) -> Any:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool | None = None) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean for {name}: {raw}")


if BaseSettings is not None:

    class Settings(BaseSettings):
        app_name: str = Field(default="FluxChi Platform API", validation_alias="FLUX_APP_NAME")
        app_version: str = Field(default="0.1.0", validation_alias="FLUX_APP_VERSION")
        api_prefix: str = Field(default="/v1", validation_alias="FLUX_API_PREFIX")
        environment: str = Field(default="development", validation_alias="FLUX_ENV")
        database_url: str | None = Field(default=None, validation_alias="FLUX_DATABASE_URL")
        secret_key: str = Field(
            default="dev-insecure-change-me",
            validation_alias="FLUX_SECRET_KEY",
        )
        access_token_ttl_sec: int = Field(
            default=3600,
            validation_alias="FLUX_ACCESS_TOKEN_TTL_SEC",
        )
        refresh_token_ttl_sec: int = Field(
            default=60 * 60 * 24 * 30,
            validation_alias="FLUX_REFRESH_TOKEN_TTL_SEC",
        )
        upload_url_ttl_sec: int = Field(
            default=900,
            validation_alias="FLUX_UPLOAD_URL_TTL_SEC",
        )
        device_limit: int = Field(default=5, validation_alias="FLUX_DEVICE_LIMIT")
        access_token_algorithm: str = Field(
            default="HS256",
            validation_alias="FLUX_ACCESS_TOKEN_ALGORITHM",
        )
        access_token_issuer: str = Field(
            default="fluxchi-platform",
            validation_alias="FLUX_ACCESS_TOKEN_ISSUER",
        )
        access_token_audience: str = Field(
            default="fluxchi-platform-api",
            validation_alias="FLUX_ACCESS_TOKEN_AUDIENCE",
        )
        allow_insecure_provider_tokens: bool | None = Field(
            default=None,
            validation_alias="FLUX_ALLOW_INSECURE_PROVIDER_TOKENS",
        )

        model_config = SettingsConfigDict(
            env_file=tuple(str(path) for path in ENV_FILES),
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
        )

        @property
        def is_production(self) -> bool:
            return self.environment.lower() == "production"

        @property
        def secret_key_is_default(self) -> bool:
            return self.secret_key == "dev-insecure-change-me"

        @property
        def insecure_provider_tokens_enabled(self) -> bool:
            if self.allow_insecure_provider_tokens is not None:
                return self.allow_insecure_provider_tokens
            return not self.is_production

else:
    from pydantic import BaseModel

    class Settings(BaseModel):
        app_name: str = "FluxChi Platform API"
        app_version: str = "0.1.0"
        api_prefix: str = "/v1"
        environment: str = _env("FLUX_ENV", "development")
        database_url: str | None = _env("FLUX_DATABASE_URL")
        secret_key: str = _env("FLUX_SECRET_KEY", "dev-insecure-change-me")
        access_token_ttl_sec: int = int(_env("FLUX_ACCESS_TOKEN_TTL_SEC", 3600))
        refresh_token_ttl_sec: int = int(_env("FLUX_REFRESH_TOKEN_TTL_SEC", 60 * 60 * 24 * 30))
        upload_url_ttl_sec: int = int(_env("FLUX_UPLOAD_URL_TTL_SEC", 900))
        device_limit: int = int(_env("FLUX_DEVICE_LIMIT", 5))
        access_token_algorithm: str = _env("FLUX_ACCESS_TOKEN_ALGORITHM", "HS256")
        access_token_issuer: str = _env("FLUX_ACCESS_TOKEN_ISSUER", "fluxchi-platform")
        access_token_audience: str = _env("FLUX_ACCESS_TOKEN_AUDIENCE", "fluxchi-platform-api")
        allow_insecure_provider_tokens: bool | None = _env_bool("FLUX_ALLOW_INSECURE_PROVIDER_TOKENS")

        @property
        def is_production(self) -> bool:
            return self.environment.lower() == "production"

        @property
        def secret_key_is_default(self) -> bool:
            return self.secret_key == "dev-insecure-change-me"

        @property
        def insecure_provider_tokens_enabled(self) -> bool:
            if self.allow_insecure_provider_tokens is not None:
                return self.allow_insecure_provider_tokens
            return not self.is_production


@lru_cache
def get_settings() -> Settings:
    if BaseSettings is not None:
        return Settings()
    return Settings()


settings = get_settings()
