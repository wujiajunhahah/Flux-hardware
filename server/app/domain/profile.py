from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProfileSummaryPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    retained_feedback_count: int = Field(ge=0)
    avg_absolute_error: float | None = None
    last_feedback_at: datetime | None = None


class UpdateProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_version: int = Field(ge=1)
    calibration_offset: float
    estimated_accuracy: float = Field(ge=0, le=100)
    training_count: int = Field(ge=0)
    active_model_release_id: str | None = None
    summary: ProfileSummaryPayload


class UpdateDeviceCalibrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_version: int = Field(ge=1)
    device_name: str = Field(min_length=1, max_length=255)
    sensor_profile: dict[str, Any] = Field(default_factory=dict)
    calibration_offset: float
