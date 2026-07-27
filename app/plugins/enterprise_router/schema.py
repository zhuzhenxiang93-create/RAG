"""Validated output schema for the generative LoRA router."""

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


QUESTION_TYPES = (
    "basic",
    "semantic",
    "intra_document_reasoning",
    "project_related",
    "constrained",
    "conflicting_info",
    "completeness",
    "miscellaneous",
    "high_level",
    "info_not_found",
)
SOURCE_TYPES = (
    "slack",
    "gmail",
    "linear",
    "google_drive",
    "hubspot",
    "fireflies",
    "github",
    "jira",
    "confluence",
)

QuestionType = Literal[
    "basic",
    "semantic",
    "intra_document_reasoning",
    "project_related",
    "constrained",
    "conflicting_info",
    "completeness",
    "miscellaneous",
    "high_level",
    "info_not_found",
]
SourceType = Literal[
    "slack",
    "gmail",
    "linear",
    "google_drive",
    "hubspot",
    "fireflies",
    "github",
    "jira",
    "confluence",
]
RetrievalDepth = Literal["shallow", "standard", "deep"]
Answerability = Literal["answerable", "unknown", "unanswerable"]


class RouteDecision(BaseModel):
    """The smallest routing contract that can change downstream retrieval."""

    question_type: QuestionType
    sources: List[SourceType] = Field(min_length=1)
    multi_document: bool
    conflict_check: bool
    completeness_required: bool
    retrieval_depth: RetrievalDepth
    answerability: Answerability

    @field_validator("sources")
    @classmethod
    def unique_sources(cls, value: List[str]) -> List[str]:
        return list(dict.fromkeys(value))

    def canonical_json(self) -> str:
        return self.model_dump_json(exclude_none=True)

