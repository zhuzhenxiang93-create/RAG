from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvidenceType = Literal["case", "statute"]


class EvidenceItemV1(BaseModel):
    """A compact, de-identified item that the generator is allowed to cite."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    evidence_type: EvidenceType
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    accusations: list[str] = Field(default_factory=list)
    relevant_articles: list[int] = Field(default_factory=list)
    penalty: dict[str, object] | None = None
    source_url: str | None = None
    source_status: str | None = None
    effective_date: str | None = None
    expiry_date: str | None = None
    legal_status: str | None = None


class EvidencePacketV1(BaseModel):
    """The complete and exclusive evidence boundary supplied to the SFT model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["evidence-packet-v1"] = "evidence-packet-v1"
    request_id: str = Field(min_length=1)
    fact: str = Field(min_length=1)
    as_of_date: str | None = None
    predicted_accusations: list[str] = Field(default_factory=list)
    sentencing_baseline: dict[str, object] | None = None
    evidence: list[EvidenceItemV1] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_evidence_ids(self) -> EvidencePacketV1:
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_id values must be unique")
        return self


class GroundedClaimV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class LegalAnalysisV1(BaseModel):
    """Strict JSON output contract for grounded legal analysis."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["legal-analysis-v1"] = "legal-analysis-v1"
    disposition: Literal["analyzed", "insufficient_evidence"]
    candidate_accusations: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    legal_basis: list[GroundedClaimV1] = Field(default_factory=list)
    analogous_cases: list[GroundedClaimV1] = Field(default_factory=list)
    sentencing_assessment: list[GroundedClaimV1] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    requires_manual_review: bool
    refusal_reason: str | None = None

    @model_validator(mode="after")
    def enforce_abstention_semantics(self) -> LegalAnalysisV1:
        if self.disposition == "insufficient_evidence":
            if not self.refusal_reason:
                raise ValueError("insufficient_evidence requires refusal_reason")
            if self.confidence != "low" or not self.requires_manual_review:
                raise ValueError("abstention must be low confidence and require manual review")
        elif self.refusal_reason is not None:
            raise ValueError("analyzed output must not include refusal_reason")
        return self
