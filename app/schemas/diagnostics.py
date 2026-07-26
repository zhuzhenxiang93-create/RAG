"""Non-sensitive runtime diagnostics."""

from typing import Optional

from pydantic import BaseModel


class DiagnosticsResponse(BaseModel):
    version: str
    mode: str
    uptime_seconds: float
    document_count: int
    parent_chunk_count: int
    child_chunk_count: int
    index_ready: bool
    index_fingerprint: Optional[str] = None
