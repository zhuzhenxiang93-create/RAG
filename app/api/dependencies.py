"""Lazy shared services."""

from fastapi import Request

from app.indexing.service import IndexService


def get_index_service(request: Request) -> IndexService:
    service = getattr(request.app.state, "index_service", None)
    if service is None:
        settings = request.app.state.settings
        service = IndexService(settings.data_dir)
        request.app.state.index_service = service
    return service
