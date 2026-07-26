"""Health response schemas."""

from typing import Literal

from pydantic import BaseModel


class CapabilityStatus(BaseModel):
    """Availability of optional runtime capabilities."""

    pdf: bool
    docx: bool
    xlsx: bool
    faiss: bool
    transformers: bool
    ocr: bool
    llm: bool


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: Literal["ok"]
    version: str
    mode: Literal["lite", "full"]
    capabilities: CapabilityStatus
