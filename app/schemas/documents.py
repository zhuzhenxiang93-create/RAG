"""Document ingestion and chunk schemas."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ContentType = Literal["text", "heading", "table", "image_ocr"]


class ParsedElement(BaseModel):
    """A location-aware element extracted from a source document."""

    content: str
    content_type: ContentType = "text"
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A retrievable child or context-bearing parent chunk."""

    chunk_id: str
    document_id: str
    level: Literal["parent", "child"]
    content: str
    parent_id: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    content_type: ContentType = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    """Persisted metadata for an ingested document."""

    document_id: str
    filename: str
    file_type: str
    size_bytes: int
    sha256: str
    status: Literal["ready", "failed"]
    element_count: int = 0
    parent_chunk_count: int = 0
    child_chunk_count: int = 0
    created_at: datetime
    error: Optional[str] = None


class DocumentDetail(DocumentRecord):
    """Document metadata with optional parsed elements and chunks."""

    elements: List[ParsedElement] = Field(default_factory=list)
    chunks: List[DocumentChunk] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    documents: List[DocumentRecord]
    total: int
