from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request

from ...core.auth import AuthPrincipal, require_principal
from ...core.errors import ok
from ...domain.feedback import CreateFeedbackEventRequest
from ...services.feedback_service import FeedbackService

router = APIRouter()
service = FeedbackService()


@router.post("/feedback-events")
def create_feedback_event(
    request: Request,
    payload: CreateFeedbackEventRequest,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)],
    principal: AuthPrincipal = Depends(require_principal),
):
    return ok(request, service.create_feedback_event(principal, payload, idempotency_key))


@router.get("/feedback-events")
def list_feedback_events(
    request: Request,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    session_id: str | None = None,
    principal: AuthPrincipal = Depends(require_principal),
):
    return ok(
        request,
        service.list_feedback_events(principal, cursor=cursor, limit=limit, session_id=session_id),
    )
