from __future__ import annotations

from ..models import CaseStep, LoadedCase, RunContext, StepResult
from . import apply_extracts, auth_headers, build_url, execute_json_request


def bootstrap(context: RunContext, _case: LoadedCase, step: CaseStep) -> StepResult:
    query = {
        "device_id": context.variables["device_id"],
        "platform": step.input["platform"],
        "channel": step.input.get("channel", "stable"),
    }
    result = execute_json_request(
        context,
        method="GET",
        url=build_url(context, "/v1/sync/bootstrap", query),
        headers=auth_headers(context),
    )
    return apply_extracts(result, step, context)
