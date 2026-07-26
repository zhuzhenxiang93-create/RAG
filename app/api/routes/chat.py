"""Trustworthy document question answering."""

from fastapi import APIRouter, Request

from app.api.dependencies import get_index_service
from app.schemas.qa import QARequest, QAResponse
from app.services.qa import QAService

router = APIRouter(tags=["qa"])


@router.post("/chat", response_model=QAResponse)
async def chat(payload: QARequest, request: Request) -> QAResponse:
    return QAService(get_index_service(request)).answer(payload)
