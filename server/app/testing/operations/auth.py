from __future__ import annotations

from ..models import CaseStep, LoadedCase, RunContext, StepResult
from . import apply_extracts, build_url, execute_json_request


def sign_up(context: RunContext, case: LoadedCase, step: CaseStep) -> StepResult:
    payload = {
        "provider": case.fixtures["provider"],
        "provider_token": case.fixtures["provider_token"],
        "device": case.fixtures["device"],
    }
    result = execute_json_request(
        context,
        method="POST",
        url=build_url(context, "/v1/auth/sign-up"),
        payload=payload,
    )
    return apply_extracts(result, step, context)


def sign_in(context: RunContext, case: LoadedCase, step: CaseStep) -> StepResult:
    payload = {
        "provider": case.fixtures["provider"],
        "provider_token": case.fixtures["provider_token"],
        "device": case.fixtures["device"],
    }
    result = execute_json_request(
        context,
        method="POST",
        url=build_url(context, "/v1/auth/sign-in"),
        payload=payload,
    )
    return apply_extracts(result, step, context)


def refresh(context: RunContext, _case: LoadedCase, step: CaseStep) -> StepResult:
    result = execute_json_request(
        context,
        method="POST",
        url=build_url(context, "/v1/auth/refresh"),
        payload={"refresh_token": context.variables["refresh_token"]},
    )
    return apply_extracts(result, step, context)


def sign_out(context: RunContext, _case: LoadedCase, step: CaseStep) -> StepResult:
    result = execute_json_request(
        context,
        method="POST",
        url=build_url(context, "/v1/auth/sign-out"),
        payload={"refresh_token": context.variables["refresh_token"]},
    )
    return apply_extracts(result, step, context)
