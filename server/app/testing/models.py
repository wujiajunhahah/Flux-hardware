from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .transport import TargetProfile, Transport, TransportRequest, TransportResponse

KNOWN_OPERATIONS = (
    "auth.refresh",
    "auth.sign_in",
    "auth.sign_out",
    "auth.sign_up",
    "feedback.create",
    "feedback.list",
    "sessions.create",
    "sessions.download",
    "sessions.finalize",
    "sessions.get",
    "sessions.upload_blob",
    "sync.bootstrap",
)

FAIL_ARTIFACT_TYPES = {"http", "domain_payloads", "blob_copy", "step_trace"}


class ExpectationRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    exists: bool | None = None
    equals: Any | None = None
    non_empty: bool | None = None
    scheme: Literal["http", "https"] | None = None

    @model_validator(mode="after")
    def validate_rule(self) -> "ExpectationRule":
        if not any(
            [
                self.exists is not None,
                self.equals is not None,
                self.non_empty is not None,
                self.scheme is not None,
            ]
        ):
            raise ValueError("expectation rule must define at least one matcher")
        return self


class StepExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool | None = None
    status: int | None = Field(default=None, ge=100, le=599)
    json_paths: list[ExpectationRule] = Field(default_factory=list)
    blob_fixture_ref: str | None = None


class ExtractRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    path: str = Field(min_length=1)


class ArtifactPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    on_fail: list[str] = Field(default_factory=lambda: ["http", "domain_payloads", "blob_copy", "step_trace"])
    on_pass: list[str] = Field(default_factory=lambda: ["summary"])

    @field_validator("on_fail")
    @classmethod
    def validate_on_fail(cls, values: list[str]) -> list[str]:
        invalid = sorted({value for value in values if value not in FAIL_ARTIFACT_TYPES})
        if invalid:
            raise ValueError(f"invalid on_fail artifact types: {', '.join(invalid)}")
        return values


class RedactionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headers: list[str] = Field(default_factory=list)
    json_paths: list[str] = Field(default_factory=list)


class CaseStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)
    expect: StepExpectations = Field(default_factory=StepExpectations)
    extract: list[ExtractRule] = Field(default_factory=list)

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, value: str) -> str:
        if value not in KNOWN_OPERATIONS:
            raise ValueError(f"unknown operation: {value}")
        return value

    @field_validator("extract", mode="before")
    @classmethod
    def normalize_extract(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, dict):
            return [{"name": name, "path": path} for name, path in value.items()]
        return value


class LoadedCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    source: Literal["manual", "captured"]
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)
    target_profile: str = "staging"
    seed: dict[str, Any] = Field(default_factory=dict)
    fixtures: dict[str, Any] = Field(default_factory=dict)
    steps: list[CaseStep] = Field(min_length=1)
    artifact_policy: ArtifactPolicy = Field(default_factory=ArtifactPolicy)
    redaction: RedactionPolicy = Field(default_factory=RedactionPolicy)


class RunContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    run_id: str = Field(min_length=1)
    target_profile: TargetProfile
    transport: Transport
    variables: dict[str, Any] = Field(default_factory=dict)
    artifacts_dir: Path
    event_sink: Any | None = None


class StepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    request: TransportRequest | None = None
    response: TransportResponse | None = None
    parsed_json: Any | None = None
    extracted: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


class RunEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    seq: int = Field(ge=1)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str = Field(min_length=1)
    case_id: str | None = None
    step_id: str | None = None
    type: str = Field(min_length=1)
    status: str = Field(min_length=1)
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
