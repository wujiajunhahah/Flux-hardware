from __future__ import annotations

from ..models import CaseStep, LoadedCase, RunContext, StepResult
from . import apply_extracts, auth_headers, build_url, execute_json_request


def create(context: RunContext, case: LoadedCase, step: CaseStep) -> StepResult:
    payload = {
        **case.fixtures["feedback"],
        "device_id": context.variables["device_id"],
        "session_id": context.variables.get("session_id"),
    }
    result = execute_json_request(
        context,
        method="POST",
        url=build_url(context, "/v1/feedback-events"),
        payload=payload,
        headers=auth_headers(
            context,
            {
                "Idempotency-Key": f"{case.case_id}:{step.step_id}:{payload['feedback_event_id']}",
            },
        ),
    )
    return apply_extracts(result, step, context)


def list_feedback(context: RunContext, _case: LoadedCase, step: CaseStep) -> StepResult:
    query = {}
    if "session_id" in context.variables:
        query["session_id"] = context.variables["session_id"]
    result = execute_json_request(
        context,
        method="GET",
        url=build_url(context, "/v1/feedback-events", query or None),
        headers=auth_headers(context),
    )
    return apply_extracts(result, step, context)
