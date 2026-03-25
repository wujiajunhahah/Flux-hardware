from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CreateFeedbackEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_event_id: str = Field(min_length=1, max_length=64)
    device_id: str = Field(min_length=1, max_length=64)
    session_id: str | None = Field(default=None, min_length=1, max_length=64)
    predicted_stamina: int = Field(ge=0, le=100)
    actual_stamina: int = Field(ge=0, le=100)
    label: Literal["fatigued", "alert"]
    kss: int | None = Field(default=None, ge=1, le=9)
    note: str | None = Field(default=None, max_length=4000)
    created_at: datetime
