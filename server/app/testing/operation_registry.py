"""
V1A case-to-operation matrix

- bootstrap_happy_path_v1: auth.sign_up, auth.sign_in, sync.bootstrap
- session_upload_happy_path_v1: auth.sign_up, auth.sign_in, sessions.create, sessions.upload_blob, sessions.finalize
- feedback_event_happy_path_v1: auth.sign_up, auth.sign_in, feedback.create, feedback.list
- full_chain_happy_path_v1: auth.sign_up, auth.sign_in, sync.bootstrap, sessions.create, sessions.upload_blob, sessions.finalize, feedback.create, feedback.list
- session_upload_requires_https_url_v1: auth.sign_up, auth.sign_in, sessions.create
- feedback_event_without_session_fallback_v1: auth.sign_up, auth.sign_in, feedback.create, feedback.list

Handler ownership

- auth.* -> operations/auth.py
- sync.* -> operations/sync.py
- sessions.* -> operations/sessions.py
- feedback.* -> operations/feedback.py
"""

from __future__ import annotations

from typing import Callable

from .models import CaseStep, LoadedCase, RunContext, StepResult
from .operations import auth, feedback, sessions, sync

OperationHandler = Callable[[RunContext, LoadedCase, CaseStep], StepResult]

_OPERATIONS: dict[str, OperationHandler] = {
    "auth.sign_up": auth.sign_up,
    "auth.sign_in": auth.sign_in,
    "auth.refresh": auth.refresh,
    "auth.sign_out": auth.sign_out,
    "sync.bootstrap": sync.bootstrap,
    "sessions.create": sessions.create,
    "sessions.upload_blob": sessions.upload_blob,
    "sessions.finalize": sessions.finalize,
    "sessions.get": sessions.get_session,
    "sessions.download": sessions.download,
    "feedback.create": feedback.create,
    "feedback.list": feedback.list_feedback,
}

_MUTATING_OPERATIONS = {
    "auth.sign_up",
    "auth.sign_in",
    "auth.refresh",
    "auth.sign_out",
    "sessions.create",
    "sessions.upload_blob",
    "sessions.finalize",
    "feedback.create",
}


def get_operation_handler(operation: str) -> OperationHandler:
    try:
        return _OPERATIONS[operation]
    except KeyError as exc:
        raise KeyError(f"unknown operation handler: {operation}") from exc


def is_mutating_operation(operation: str) -> bool:
    return operation in _MUTATING_OPERATIONS


def execute_operation(context: RunContext, case: LoadedCase, step: CaseStep) -> StepResult:
    handler = get_operation_handler(step.operation)
    return handler(context, case, step)
