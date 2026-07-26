"""Trustworthy question-answering schemas."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.search import SearchResponse, SearchStrategy


EvidenceRelation = Literal["support", "contradict", "irrelevant"]
AnswerDecision = Literal["answer", "clarify", "abstain"]


class EvidenceItem(BaseModel):
    evidence_id: str
    document_id: str
    filename: str
    chunk_id: str
    page: Optional[int] = None
    section: Optional[str] = None
    text: str
    relation: EvidenceRelation
    relevance: float = Field(ge=0.0, le=1.0)


class ConflictItem(BaseModel):
    conflict_id: str
    evidence_ids: List[str]
    description: str
    conflict_type: Literal["numeric", "negation", "version"]


class ConfidenceBreakdown(BaseModel):
    evidence_relevance: float = Field(ge=0.0, le=1.0)
    channel_agreement: float = Field(ge=0.0, le=1.0)
    ranking_margin: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    conflict_penalty: float = Field(ge=0.0, le=1.0)


class QARequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    strategy: SearchStrategy = "adaptive"
    top_k: int = Field(default=5, ge=1, le=20)


class QAResponse(BaseModel):
    answer: str
    decision: AnswerDecision
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_breakdown: ConfidenceBreakdown
    citations: List[EvidenceItem]
    conflicts: List[ConflictItem]
    reason: str
    retrieval: SearchResponse
