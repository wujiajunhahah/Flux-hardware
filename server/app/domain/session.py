from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=64)
    device_id: str = Field(min_length=1, max_length=64)
    source: str = Field(min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=255)
    started_at: datetime
    ended_at: datetime
    duration_sec: int = Field(ge=0)
    snapshot_count: int = Field(ge=0)
    schema_version: int = Field(ge=1)
    content_type: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=1, max_length=128)


class SessionBlobPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_key: str = Field(min_length=1, max_length=1024)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=1, max_length=128)


class FinalizeSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ready"]
    blob: SessionBlobPayload
