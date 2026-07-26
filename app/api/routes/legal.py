"""Optional legal plugin endpoints."""

from fastapi import APIRouter, Request

from app.core.exceptions import AppError
from app.plugins.legal.classifier import LegalClassifier
from app.schemas.legal import (
    LegalClassificationRequest,
    LegalClassificationResponse,
    LegalPluginStatus,
)

router = APIRouter(prefix="/plugins/legal", tags=["legal-plugin"])


def _classifier(request: Request) -> LegalClassifier:
    classifier = getattr(request.app.state, "legal_classifier", None)
    if classifier is None:
        classifier = LegalClassifier(request.app.state.settings)
        request.app.state.legal_classifier = classifier
    return classifier


@router.get("/status", response_model=LegalPluginStatus)
async def legal_status(request: Request) -> LegalPluginStatus:
    return _classifier(request).status()


@router.post("/classify", response_model=LegalClassificationResponse)
async def classify(
    payload: LegalClassificationRequest, request: Request
) -> LegalClassificationResponse:
    classifier = _classifier(request)
    status = classifier.status()
    if status.state not in {"ready", "loaded"}:
        raise AppError(
            status.message,
            code="legal_plugin_unavailable",
            status_code=503,
        )
    try:
        return classifier.predict(payload.text, payload.top_k)
    except RuntimeError as exc:
        raise AppError(
            str(exc),
            code="legal_inference_failed",
            status_code=503,
        ) from exc
