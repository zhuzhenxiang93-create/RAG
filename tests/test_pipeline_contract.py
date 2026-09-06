from __future__ import annotations

import pytest
from pydantic import ValidationError

from legalmind.generation.schemas import StructuredLegalAnalysis
from legalmind.pipeline.contracts import LegalCaseAnalysisResponse
from legalmind.sentencing.schemas import FinePrediction, SentencePrediction, SentenceType


def minimal_response() -> dict:
    return {
        "classification": {"status": "degraded_no_classifier", "labels": []},
        "retrieval": {
            "status": "degraded_no_index",
            "evidence": [],
            "statute_status": "degraded_no_statute_index",
            "statutes": [],
        },
        "analysis": StructuredLegalAnalysis(
            analysis="证据不足，需要人工复核。",
            confidence="low",
            requires_manual_review=True,
        ).model_dump(),
        "citation_validation": {"valid": True},
        "evidence_firewall": {
            "passed": False,
            "checks": {},
            "blocking_reasons": ["case_evidence_present"],
            "rejected_statutes": [],
            "requires_manual_review": True,
        },
        "sentencing": {"status": "degraded_no_sentencing_model"},
        "requires_manual_review": True,
        "disclaimer": "仅用于研究，不构成法律意见。",
    }


def test_contract_forbids_unknown_top_level_fields() -> None:
    value = minimal_response()
    value["unexpected"] = True
    with pytest.raises(ValidationError):
        LegalCaseAnalysisResponse.model_validate(value)


def test_contract_forbids_unknown_nested_fields() -> None:
    value = minimal_response()
    value["classification"]["model_internal_state"] = "must-not-leak"
    with pytest.raises(ValidationError):
        LegalCaseAnalysisResponse.model_validate(value)


def test_contract_forbids_citations_outside_retrieval() -> None:
    value = minimal_response()
    value["analysis"]["similar_cases"] = [{"case_id": "CASE-404", "reason": "不存在"}]
    with pytest.raises(ValidationError, match="outside retrieval evidence"):
        LegalCaseAnalysisResponse.model_validate(value)


def test_contract_cannot_hide_manual_review_requirement() -> None:
    value = minimal_response()
    value["requires_manual_review"] = False
    with pytest.raises(ValidationError, match="cannot be false"):
        LegalCaseAnalysisResponse.model_validate(value)


def test_contract_json_round_trip_preserves_version() -> None:
    response = LegalCaseAnalysisResponse.model_validate(minimal_response())
    restored = LegalCaseAnalysisResponse.model_validate_json(response.model_dump_json())
    assert restored == response
    assert restored.schema_version == "legal-case-analysis-v1"


@pytest.mark.parametrize("sentence_type", [SentenceType.death, SentenceType.life])
def test_non_month_sentences_reject_fake_months(sentence_type: SentenceType) -> None:
    with pytest.raises(ValidationError, match="must not use imprisonment months"):
        SentencePrediction(sentence_type=sentence_type, imprisonment_months=999)


def test_fine_amount_requires_explicit_positive_prediction() -> None:
    with pytest.raises(ValidationError, match="requires imposed=true"):
        FinePrediction(imposed=None, amount=1000)


def test_fine_zero_is_not_silently_treated_as_missing() -> None:
    prediction = FinePrediction(imposed=True, amount=0)
    assert prediction.imposed is True
    assert prediction.amount == 0
