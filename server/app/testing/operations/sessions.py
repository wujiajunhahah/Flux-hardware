from __future__ import annotations

from ..models import CaseStep, LoadedCase, RunContext, StepResult
from . import (
    apply_extracts,
    auth_headers,
    build_url,
    encode_session_blob,
    execute_bytes_request,
    execute_json_request,
)


def create(context: RunContext, case: LoadedCase, step: CaseStep) -> StepResult:
    blob_bytes, blob_sha256 = encode_session_blob(case)
    payload = {
        **step.input,
        "device_id": context.variables["device_id"],
        "size_bytes": len(blob_bytes),
        "sha256": blob_sha256,
    }
    result = execute_json_request(
        context,
        method="POST",
        url=build_url(context, "/v1/sessions"),
        payload=payload,
        headers=auth_headers(
            context,
            {
                "Idempotency-Key": f"{case.case_id}:{step.step_id}:{payload['session_id']}",
            },
        ),
    )
    context.variables["session_blob_bytes"] = blob_bytes
    context.variables["session_blob_sha256"] = blob_sha256
    context.variables["session_blob_size_bytes"] = len(blob_bytes)
    return apply_extracts(result, step, context)


def upload_blob(context: RunContext, _case: LoadedCase, step: CaseStep) -> StepResult:
    result = execute_bytes_request(
        context,
        method="PUT",
        url=context.variables["upload_url"],
        payload=context.variables["session_blob_bytes"],
        headers={"Content-Type": "application/json"},
    )
    return apply_extracts(result, step, context)


def finalize(context: RunContext, _case: LoadedCase, step: CaseStep) -> StepResult:
    payload = {
        "status": step.input.get("status", "ready"),
        "blob": {
            "object_key": context.variables["object_key"],
            "size_bytes": context.variables["session_blob_size_bytes"],
            "sha256": context.variables["session_blob_sha256"],
        },
    }
    result = execute_json_request(
        context,
        method="PATCH",
        url=build_url(context, f"/v1/sessions/{context.variables['session_id']}"),
        payload=payload,
        headers=auth_headers(context),
    )
    return apply_extracts(result, step, context)


def get_session(context: RunContext, _case: LoadedCase, step: CaseStep) -> StepResult:
    result = execute_json_request(
        context,
        method="GET",
        url=build_url(context, f"/v1/sessions/{context.variables['session_id']}"),
        headers=auth_headers(context),
    )
    return apply_extracts(result, step, context)


def download(context: RunContext, _case: LoadedCase, step: CaseStep) -> StepResult:
    meta_result = execute_json_request(
        context,
        method="GET",
        url=build_url(context, f"/v1/sessions/{context.variables['session_id']}/download"),
        headers=auth_headers(context),
    )
    download_url = meta_result.parsed_json["data"]["download_url"]
    context.variables["download_url"] = download_url
    blob_result = execute_bytes_request(
        context,
        method="GET",
        url=download_url,
        headers=context.target_profile.default_headers,
    )
    blob_result.extracted.update({"download_url": download_url})
    return apply_extracts(blob_result, step, context)
