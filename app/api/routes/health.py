"""Health and capability endpoints."""

from importlib.util import find_spec

from fastapi import APIRouter, Request

from app import __version__
from app.schemas.health import CapabilityStatus, HealthResponse

router = APIRouter(tags=["system"])


def _available(module: str) -> bool:
    return find_spec(module) is not None


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return process health without initializing optional heavy providers."""
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        version=__version__,
        mode=settings.app_mode,
        capabilities=CapabilityStatus(
            pdf=_available("pypdf") or _available("PyPDF2"),
            docx=_available("docx"),
            xlsx=_available("openpyxl"),
            faiss=_available("faiss"),
            transformers=_available("transformers"),
            ocr=settings.ocr_enabled,
            llm=bool(settings.llm_api_key),
        ),
    )
