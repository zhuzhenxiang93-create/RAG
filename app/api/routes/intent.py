"""General intent classification and routing endpoints."""

from fastapi import APIRouter, Request

from app.api.dependencies import get_intent_classifier
from app.core.exceptions import AppError
from app.schemas.intent import (
    IntentClassificationRequest,
    IntentClassificationResponse,
    IntentPluginStatus,
)

router = APIRouter(prefix="/plugins/intent", tags=["intent-routing"])


@router.get("/status", response_model=IntentPluginStatus)
async def intent_status(request: Request) -> IntentPluginStatus:
    return get_intent_classifier(request).status()


@router.post("/classify", response_model=IntentClassificationResponse)
async def classify(
    payload: IntentClassificationRequest, request: Request
) -> IntentClassificationResponse:
    classifier = get_intent_classifier(request)
    status = classifier.status()
    if status.state not in {"lite_ready", "ready", "loaded"}:
        raise AppError(
            status.message,
            code="intent_plugin_unavailable",
            status_code=503,
        )
    try:
        return classifier.predict(payload.text, payload.top_k)
    except RuntimeError as exc:
        raise AppError(
            str(exc),
            code="intent_inference_failed",
            status_code=503,
        ) from exc

