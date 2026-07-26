"""Schemas for the local interview demo workflow."""

from typing import List, Literal

from pydantic import BaseModel

from app.schemas.search import IndexBuildResponse


class DemoBootstrapResponse(BaseModel):
    status: Literal["ready"]
    added: List[str]
    skipped: List[str]
    index: IndexBuildResponse
