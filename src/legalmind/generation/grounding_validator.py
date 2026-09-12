from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from legalmind.data.privacy import scan_privacy
from legalmind.generation.contracts_v2 import EvidencePacketV1, LegalAnalysisV1
from legalmind.generation.structured import extract_json_object
from legalmind.retrieval.temporal import assess_statute_hit
from legalmind.schemas import SearchHit


def parse_legal_analysis(value: str) -> LegalAnalysisV1:
    return LegalAnalysisV1.model_validate_json(extract_json_object(value))


def _claims(analysis: LegalAnalysisV1):
    yield from analysis.legal_basis
    yield from analysis.analogous_cases
    yield from analysis.sentencing_assessment


def validate_grounded_analysis(
    analysis: LegalAnalysisV1,
    packet: EvidencePacketV1,
    *,
    allow_synthetic_test_evidence: bool = False,
) -> dict[str, Any]:
    catalog = {item.evidence_id: item for item in packet.evidence}
    cited = {evidence_id for claim in _claims(analysis) for evidence_id in claim.evidence_ids}
    unknown = sorted(cited - catalog.keys())
    wrong_type: list[str] = []
    temporal_or_source_errors: dict[str, list[str]] = {}
    for claim in analysis.legal_basis:
        for evidence_id in claim.evidence_ids:
            item = catalog.get(evidence_id)
            if item is None:
                continue
            if item.evidence_type != "statute":
                wrong_type.append(evidence_id)
                continue
            synthetic = (
                allow_synthetic_test_evidence and item.source_status == "synthetic_test_only"
            )
            if synthetic:
                continue
            hit = SearchHit(
                chunk_id=item.evidence_id,
                case_id=item.title,
                text=item.summary,
                score=1.0,
                evidence_type="statute",
                source_url=item.source_url,
                effective_date=item.effective_date,
                expiry_date=item.expiry_date,
                legal_status=item.legal_status,
                source_status=item.source_status,
            )
            reasons = assess_statute_hit(hit, packet.as_of_date)
            if reasons:
                temporal_or_source_errors[evidence_id] = reasons
    for claim in analysis.analogous_cases:
        for evidence_id in claim.evidence_ids:
            item = catalog.get(evidence_id)
            if item is not None and item.evidence_type != "case":
                wrong_type.append(evidence_id)

    privacy_counts = scan_privacy(analysis.model_dump_json())
    privacy_hits = sum(privacy_counts.values())
    has_claims = any(True for _ in _claims(analysis))
    analyzed_without_support = analysis.disposition == "analyzed" and not has_claims
    valid = not (
        unknown
        or wrong_type
        or temporal_or_source_errors
        or privacy_hits
        or analyzed_without_support
    )
    return {
        "valid": valid,
        "unknown_evidence_ids": unknown,
        "wrong_evidence_type_ids": sorted(set(wrong_type)),
        "statute_validity_errors": temporal_or_source_errors,
        "privacy_hits": privacy_counts,
        "analyzed_without_support": analyzed_without_support,
        "claim_coverage": 1.0
        if has_claims or analysis.disposition == "insufficient_evidence"
        else 0.0,
    }


def validate_generated_text(value: str, packet: EvidencePacketV1) -> dict[str, Any]:
    try:
        analysis = parse_legal_analysis(value)
    except (ValueError, ValidationError) as error:
        return {"valid": False, "schema_valid": False, "error": str(error)}
    report = validate_grounded_analysis(analysis, packet)
    return {"schema_valid": True, "analysis": analysis, **report}
