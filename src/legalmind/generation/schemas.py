from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictGenerationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimilarCaseCitation(StrictGenerationModel):
    case_id: str
    reason: str


class StructuredLegalAnalysis(StrictGenerationModel):
    predicted_accusations: list[str] = Field(default_factory=list)
    relevant_articles: list[int] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    similar_cases: list[SimilarCaseCitation] = Field(default_factory=list)
    analysis: str
    confidence: Literal["high", "medium", "low"]
    requires_manual_review: bool


class SimplifiedLegalAnalysis(StrictGenerationModel):
    candidate_accusations: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    requires_manual_review: bool
