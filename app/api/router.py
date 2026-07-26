"""Top-level API router."""

from fastapi import APIRouter

from app.api.routes.chat import router as chat_router
from app.api.routes.demo import router as demo_router
from app.api.routes.diagnostics import router as diagnostics_router
from app.api.routes.documents import router as documents_router
from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.health import router as health_router
from app.api.routes.legal import router as legal_router
from app.api.routes.search import router as search_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(diagnostics_router)
api_router.include_router(demo_router)
api_router.include_router(documents_router)
api_router.include_router(search_router)
api_router.include_router(chat_router)
api_router.include_router(legal_router)
api_router.include_router(evaluation_router)
