"""Index build and explainable search endpoints."""

from fastapi import APIRouter, Request

from app.api.dependencies import get_index_service
from app.schemas.search import IndexBuildResponse, SearchRequest, SearchResponse

router = APIRouter(tags=["retrieval"])


@router.post("/index/build", response_model=IndexBuildResponse)
async def build_index(request: Request) -> IndexBuildResponse:
    return get_index_service(request).build()


@router.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest, request: Request) -> SearchResponse:
    return get_index_service(request).search(payload)
