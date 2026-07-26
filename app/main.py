"""FastAPI application factory."""

from time import perf_counter
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.request_tracking import register_request_tracking


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create an application without loading models or creating indexes."""
    active_settings = settings or get_settings()
    configure_logging(active_settings.log_level, active_settings.log_format)

    app = FastAPI(
        title="DocMind-RAG",
        description="Adaptive retrieval and trustworthy QA for complex documents.",
        version=__version__,
    )
    app.state.settings = active_settings
    app.state.started_at = perf_counter()
    register_request_tracking(app)
    register_exception_handlers(app)
    app.include_router(api_router, prefix=active_settings.api_prefix)

    frontend_dir = active_settings.project_root / "frontend"
    if frontend_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def redirect_to_ui() -> RedirectResponse:
            return RedirectResponse(url="/ui/")
    return app


app = create_app()
