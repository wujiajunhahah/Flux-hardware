from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.responses import FileResponse

from ...core.auth import AuthPrincipal, require_principal
from ...core.errors import ok
from ...domain.session import CreateSessionRequest, FinalizeSessionRequest
from ...services.session_service import SessionService

router = APIRouter()
service = SessionService()


@router.post("/sessions")
def create_session(
    request: Request,
    payload: CreateSessionRequest,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)],
    principal: AuthPrincipal = Depends(require_principal),
):
    data = service.create_session(principal, payload, idempotency_key)
    upload = data.get("upload")
    if isinstance(upload, dict) and "upload_token" in upload:
        token = upload.pop("upload_token")
        upload["upload_url"] = str(
            request.url_for("put_session_blob", session_id=data["session"]["session_id"]).include_query_params(token=token)
        )
    session = data.get("session")
    if isinstance(session, dict) and "download_token" in session:
        session.pop("download_token")
        session["download_url"] = str(request.url_for("get_session_download_url", session_id=session["session_id"]))
    return ok(request, data)


@router.put("/session-blobs/{session_id}", include_in_schema=False, name="put_session_blob")
async def put_session_blob(
    session_id: str,
    request: Request,
    token: Annotated[str, Query(min_length=1)],
):
    payload = await request.body()
    service.upload_session_blob(
        session_id=session_id,
        token=token,
        payload=payload,
        content_type=request.headers.get("content-type"),
    )
    return Response(status_code=200)


@router.patch("/sessions/{session_id}")
def finalize_session(
    session_id: str,
    request: Request,
    payload: FinalizeSessionRequest,
    principal: AuthPrincipal = Depends(require_principal),
):
    data = service.finalize_session(principal, session_id, payload)
    data["session"].pop("download_token")
    data["session"]["download_url"] = str(request.url_for("get_session_download_url", session_id=session_id))
    return ok(request, data)


@router.get("/sessions")
def list_sessions(
    request: Request,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    status: Literal["pending_upload", "ready", "failed", "deleted"] | None = None,
    principal: AuthPrincipal = Depends(require_principal),
):
    return ok(
        request,
        service.list_sessions(principal, cursor=cursor, limit=limit, status=status),
    )


@router.get("/sessions/{session_id}")
def get_session(
    session_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(require_principal),
):
    return ok(request, service.get_session(principal, session_id))


@router.get("/sessions/{session_id}/download")
def get_session_download_url(
    session_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(require_principal),
):
    data = service.get_download_url(principal, session_id)
    token = data.pop("download_token")
    data["download_url"] = str(
        request.url_for("get_session_blob", session_id=session_id).include_query_params(token=token)
    )
    return ok(request, data)


@router.get("/session-blobs/{session_id}", include_in_schema=False, name="get_session_blob")
def get_session_blob(session_id: str, token: Annotated[str, Query(min_length=1)]):
    download = service.download_session_blob(session_id=session_id, token=token)
    return FileResponse(
        path=str(download["path"]),
        media_type=download["content_type"],
        filename=download["filename"],
    )
