from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CaseRecord(BaseModel):
    case_id: str
    fact: str
    accusations: list[str] = Field(default_factory=list)
    accusation_ids: list[int] = Field(default_factory=list)
    relevant_articles: list[int] = Field(default_factory=list)
    penalty: dict | None = None
    source_split: str | None = None

    @field_validator("fact")
    @classmethod
    def non_empty_fact(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("fact must not be empty")
        return value


class CaseChunk(BaseModel):
    chunk_id: str
    case_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    accusations: list[str] = Field(default_factory=list)
    relevant_articles: list[int] = Field(default_factory=list)
    source_split: str | None = None
    parent_title: str | None = None
    text_sha256: str | None = None
    evidence_type: str = "case"
    source_url: str | None = None
    promulgation_date: str | None = None
    effective_date: str | None = None
    expiry_date: str | None = None
    legal_status: str | None = None
    source_status: str | None = None
    retrieved_at: str | None = None
    checksum: str | None = None


class StatuteChunk(BaseModel):
    statute_id: str
    clause_id: str
    law_name: str
    article_number: int
    title: str
    text: str
    source_url: str
    version: str
    promulgation_date: str | None = None
    effective_from: str | None = None
    expiry_date: str | None = None
    status: str = "unknown"
    retrieved_at: str | None = None
    checksum: str | None = None
    verified_at: str | None = None
    source_status: str = "unverified_source"


class LabelScore(BaseModel):
    label_id: int
    label: str
    probability: float


class ClassificationResult(BaseModel):
    labels: list[LabelScore]
    thresholds: dict[int, float] = Field(default_factory=dict)
    used_fallback: bool = False
    max_probability: float | None = None


class SearchHit(BaseModel):
    chunk_id: str
    case_id: str
    text: str
    score: float
    source_scores: dict[str, float] = Field(default_factory=dict)
    accusations: list[str] = Field(default_factory=list)
    relevant_articles: list[int] = Field(default_factory=list)
    evidence_type: str = "case"
    source_url: str | None = None
    promulgation_date: str | None = None
    effective_date: str | None = None
    expiry_date: str | None = None
    legal_status: str | None = None
    source_status: str | None = None
    retrieved_at: str | None = None
    checksum: str | None = None
