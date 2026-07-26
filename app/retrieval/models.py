"""Internal retrieval data structures."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IndexedChunk:
    document_id: str
    filename: str
    chunk_id: str
    content: str
    content_type: str
    parent_id: Optional[str] = None
    parent_content: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None


@dataclass
class Candidate:
    chunk: IndexedChunk
    score: float = 0.0
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    fused_rank: Optional[int] = None
    channels: List[str] = field(default_factory=list)
