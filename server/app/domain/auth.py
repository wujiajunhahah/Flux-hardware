from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DevicePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_device_key: str = Field(min_length=1, max_length=255)
    platform: Literal["ios", "android", "web"]
    device_name: str = Field(min_length=1, max_length=255)
    app_version: str | None = Field(default=None, max_length=64)
    os_version: str | None = Field(default=None, max_length=64)


class SignInRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=64)
    provider_token: str = Field(min_length=1)
    device: DevicePayload


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str = Field(min_length=1)


class SignOutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str = Field(min_length=1)
