"""Safe, idempotent bootstrap endpoint for the local Lite demo."""

import hashlib

from fastapi import APIRouter, Request

from app.api.dependencies import get_index_service
from app.core.exceptions import AppError
from app.schemas.demo import DemoBootstrapResponse
from app.services.documents import DocumentService

router = APIRouter(prefix="/demo", tags=["demo"])


@router.post("/bootstrap", response_model=DemoBootstrapResponse)
async def bootstrap_demo(request: Request) -> DemoBootstrapResponse:
    settings = request.app.state.settings
    if settings.app_mode != "lite":
        raise AppError(
            "Demo bootstrap is available only in Lite mode",
            code="demo_disabled",
            status_code=403,
        )

    corpus_dir = settings.project_root / "data" / "eval" / "lite_v1" / "corpus"
    sources = sorted(corpus_dir.glob("*.md"))
    if not sources:
        raise AppError(
            "Demo corpus is unavailable",
            code="demo_corpus_missing",
            status_code=503,
        )

    service = DocumentService(settings.data_dir, settings.max_upload_mb)
    known_hashes = {item.sha256 for item in service.list_documents()}
    added = []
    skipped = []
    for source in sources:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in known_hashes:
            skipped.append(source.name)
            continue
        with source.open("rb") as stream:
            service.ingest(source.name, stream)
        known_hashes.add(digest)
        added.append(source.name)

    index = get_index_service(request).build()
    return DemoBootstrapResponse(
        status="ready",
        added=added,
        skipped=skipped,
        index=index,
    )
