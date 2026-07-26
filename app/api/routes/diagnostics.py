"""Operational diagnostics without secrets or host paths."""

from time import perf_counter

from fastapi import APIRouter, Request

from app import __version__
from app.schemas.diagnostics import DiagnosticsResponse
from app.services.documents import DocumentService

router = APIRouter(tags=["system"])


@router.get("/diagnostics", response_model=DiagnosticsResponse)
async def diagnostics(request: Request) -> DiagnosticsResponse:
    settings = request.app.state.settings
    documents = DocumentService(
        settings.data_dir, settings.max_upload_mb
    ).list_documents()
    index_service = getattr(request.app.state, "index_service", None)
    fingerprint = getattr(index_service, "fingerprint", "")
    return DiagnosticsResponse(
        version=__version__,
        mode=settings.app_mode,
        uptime_seconds=round(perf_counter() - request.app.state.started_at, 3),
        document_count=len(documents),
        parent_chunk_count=sum(item.parent_chunk_count for item in documents),
        child_chunk_count=sum(item.child_chunk_count for item in documents),
        index_ready=bool(fingerprint),
        index_fingerprint=fingerprint or None,
    )
