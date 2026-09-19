from __future__ import annotations

from legalmind.generation.evidence_packet import build_evidence_packet
from legalmind.generation.sft_data import build_sft_record
from legalmind.generation.simple_sft import build_simple_sft_record
from legalmind.generation.structured import (
    deterministic_grounded_analysis,
    extract_json_object,
    parse_and_validate_json,
    parse_simplified_json,
    validate_citations,
)
from legalmind.schemas import SearchHit


def hit() -> SearchHit:
    return SearchHit(
        chunk_id="case-1:c0",
        case_id="case-1",
        text="案件事实",
        score=1.0,
        relevant_articles=[234],
    )


def test_deterministic_generation_is_schema_valid() -> None:
    value = deterministic_grounded_analysis("被告人持刀致人轻伤。", ["故意伤害"], [hit()], [234])
    parsed = parse_and_validate_json(value.model_dump_json())
    assert parsed.predicted_accusations == ["故意伤害"]


def test_invalid_citation_is_detected() -> None:
    value = deterministic_grounded_analysis("事实", ["故意伤害"], [hit()], [234])
    report = validate_citations(value, set(), set())
    assert not report["valid"]


def test_sft_record_preserves_source_split_and_review_status() -> None:
    row = {
        "case_id": "c1",
        "source_split": "train",
        "fact": "案件事实。",
        "labels": {"accusations": ["盗窃"], "relevant_articles": []},
    }
    record = build_sft_record(row)
    assert record["source_split"] == "train"
    assert record["review_status"] == "unreviewed"


def test_json_object_extraction_ignores_fences_and_extra_text() -> None:
    value = '说明\n```json\n{"candidate_accusations":[]}\n```\n结束'
    assert extract_json_object(value) == '{"candidate_accusations":[]}'


def test_simplified_schema_can_canonicalize_observed_chinese_fields() -> None:
    value = (
        '{"罪名候选":["盗窃罪"],"关键事实":["秘密取得手机"],'
        '"置信度":"中等","人工复核标记":"需人工复核"}'
    )
    parsed = parse_simplified_json(value, repair=True)
    assert parsed.candidate_accusations == ["盗窃"]
    assert parsed.confidence == "medium"
    assert parsed.requires_manual_review


def test_truncated_json_is_rejected() -> None:
    try:
        extract_json_object('{"candidate_accusations":[')
    except ValueError as error:
        assert "truncated" in str(error)
    else:
        raise AssertionError("truncated JSON must fail")


def test_evidence_packet_preserves_deidentified_case_penalty_context() -> None:
    case = SearchHit(
        chunk_id="case-1:c0",
        case_id="case-1",
        text="被告人张三，盗窃后退赔。",
        score=0.9,
        accusations=["盗窃"],
        penalty={"sentence_type": "fixed_term", "imprisonment_months": 8},
    )
    packet = build_evidence_packet(
        "匿名案件事实",
        ["盗窃"],
        [case],
        [],
        as_of_date="2025-01-01",
        sentencing_baseline={"imprisonment_months": 9},
    )
    assert packet.evidence[0].penalty["imprisonment_months"] == 8
    assert packet.evidence[0].accusations == ["盗窃"]
    assert "张三" not in packet.evidence[0].summary


def test_insufficient_information_target_abstains() -> None:
    row = {
        "case_id": "case-1",
        "source_split": "train",
        "fact": "被告人与他人发生争执。后续行为和损害结果没有提供。",
        "labels": {"accusations": ["故意伤害"]},
        "cleaning_metadata": {"fact_sha256": "source-hash"},
    }
    record = build_simple_sft_record(
        row, construction_type="deterministic_insufficient_information"
    )
    assert record["target"]["candidate_accusations"] == []
    assert record["target"]["requires_manual_review"] is True
    assert record["target"]["confidence"] == "low"
    assert record["review_status"] == "unreviewed"
