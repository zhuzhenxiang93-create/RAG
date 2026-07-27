"""Search, routing and index API schemas."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


QueryType = Literal[
    "exact_lookup",
    "semantic_qa",
    "table_qa",
    "comparison",
    "summarization",
    "out_of_scope",
]
SearchStrategy = Literal["bm25", "dense", "rrf", "rrf_rerank", "adaptive"]


class RetrievalPlan(BaseModel):
    query_type: QueryType
    confidence: float = Field(ge=0.0, le=1.0)
    keywords: List[str] = Field(default_factory=list)
    bm25_weight: float = Field(ge=0.0)
    dense_weight: float = Field(ge=0.0)
    use_reranker: bool
    reason: str
    intent: Optional[str] = None
    intent_domain: Optional[str] = None
    intent_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    knowledge_base: Optional[str] = None
    routing_backend: Optional[Literal["lite", "lora"]] = None
    routing_abstained: bool = False


class SearchRequest(BaseModel):
    query: str
    strategy: SearchStrategy = "adaptive"
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        if len(normalized) > 2000:
            raise ValueError("query must not exceed 2000 characters")
        return normalized


class SearchHit(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    parent_id: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    content_type: str
    content: str
    parent_content: Optional[str] = None
    score: float
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    fused_rank: Optional[int] = None
    final_rank: int
    channels: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    strategy: SearchStrategy
    plan: RetrievalPlan
    index_fingerprint: str
    total_candidates: int
    results: List[SearchHit]
    timings_ms: Dict[str, float]


class IndexBuildResponse(BaseModel):
    status: Literal["ready"]
    fingerprint: str
    document_count: int
    child_chunk_count: int
    parent_chunk_count: int
    build_ms: float
