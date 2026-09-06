from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from legalmind.data.privacy import scan_privacy
from legalmind.generation.schemas import StructuredLegalAnalysis
from legalmind.sentencing.schemas import SentencingResponse


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LabelScorePayload(StrictModel):
    label_id: int = Field(ge=0)
    label: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)


class LegalCaseAnalysisRequest(StrictModel):
    fact: str = Field(min_length=1)
    accusations: list[str] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1, le=3)
    as_of_date: str | None = None


class ClassificationSection(StrictModel):
    status: Literal["ok", "provided", "uncertain_below_threshold", "degraded_no_classifier"]
    labels: list[LabelScorePayload] = Field(default_factory=list)
    thresholds: dict[int, float] = Field(default_factory=dict)
    used_fallback: bool = False
    max_probability: float | None = Field(default=None, ge=0.0, le=1.0)


class EvidencePayload(StrictModel):
    chunk_id: str
    case_id: str
    text: str
    score: float
    source_scores: dict[str, float] = Field(default_factory=dict)
    accusations: list[str] = Field(default_factory=list)
    relevant_articles: list[int] = Field(default_factory=list)
    evidence_type: Literal["case", "statute"]
    source_url: str | None = None
    promulgation_date: str | None = None
    effective_date: str | None = None
    expiry_date: str | None = None
    legal_status: str | None = None
    source_status: str | None = None
    retrieved_at: str | None = None
    checksum: str | None = None
    privacy_redaction_counts: dict[str, int] = Field(default_factory=dict)
    presentation: Literal["deidentified_summary", "official_statute_text"]


class RetrievalSection(StrictModel):
    status: Literal["ok", "no_match", "degraded_no_index"]
    evidence: list[EvidencePayload] = Field(default_factory=list)
    statute_status: Literal["ok", "degraded_no_statute_index", "no_verified_applicable_statute"]
    statutes: list[EvidencePayload] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_types_are_separated(self) -> RetrievalSection:
        if any(item.evidence_type != "case" for item in self.evidence):
            raise ValueError("retrieval.evidence may contain only case evidence")
        if any(item.evidence_type != "statute" for item in self.statutes):
            raise ValueError("retrieval.statutes may contain only statute evidence")
        return self


class CitationValidation(StrictModel):
    valid: bool
    invalid_case_ids: list[str] = Field(default_factory=list)
    invalid_articles: list[str] = Field(default_factory=list)


class RejectedStatute(StrictModel):
    clause_id: str
    reasons: list[str] = Field(default_factory=list)


class EvidenceFirewall(StrictModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    blocking_reasons: list[str] = Field(default_factory=list)
    rejected_statutes: list[RejectedStatute] = Field(default_factory=list)
    requires_manual_review: bool


class LegalCaseAnalysisResponse(StrictModel):
    """Versioned deterministic API response; no generative model is required."""

    schema_version: Literal["legal-case-analysis-v1"] = "legal-case-analysis-v1"
    classification: ClassificationSection
    retrieval: RetrievalSection
    analysis: StructuredLegalAnalysis
    citation_validation: CitationValidation
    evidence_firewall: EvidenceFirewall
    sentencing: SentencingResponse
    legal_as_of_date: str | None = None
    initialization_warnings: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    requires_manual_review: bool = True
    disclaimer: str

    @model_validator(mode="after")
    def enforce_cross_field_safety(self) -> LegalCaseAnalysisResponse:
        case_ids = {item.case_id for item in self.retrieval.evidence}
        cited_case_ids = {item.case_id for item in self.analysis.similar_cases}
        if not cited_case_ids.issubset(case_ids):
            raise ValueError("analysis cites a case outside retrieval evidence")

        statute_articles = {
            article for item in self.retrieval.statutes for article in item.relevant_articles
        }
        if not set(self.analysis.relevant_articles).issubset(statute_articles):
            raise ValueError("analysis cites an article outside statute evidence")

        for item in self.retrieval.evidence:
            if sum(scan_privacy(item.text).values()):
                raise ValueError(f"case evidence contains a direct identifier: {item.chunk_id}")

        if self.evidence_firewall.passed:
            if not self.citation_validation.valid:
                raise ValueError("firewall cannot pass when citation validation fails")
            if self.legal_as_of_date is None or not self.retrieval.statutes:
                raise ValueError("firewall requires an as-of date and applicable statutes")

        review_required = (
            self.classification.status not in {"ok", "provided"}
            or self.analysis.requires_manual_review
            or self.sentencing.requires_manual_review
            or self.evidence_firewall.requires_manual_review
            or not self.citation_validation.valid
        )
        if review_required and not self.requires_manual_review:
            raise ValueError(
                "requires_manual_review cannot be false while a component requires review"
            )
        if any(value < 0 for value in self.timings.values()):
            raise ValueError("timings must be non-negative")
        return self
