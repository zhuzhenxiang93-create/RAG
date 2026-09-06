from __future__ import annotations

import pytest

from legalmind.retrieval.temporal import assess_statute_hit, filter_effective_statutes
from legalmind.schemas import SearchHit


def statute(**overrides) -> SearchHit:
    values = {
        "chunk_id": "PRC-LAW#1",
        "case_id": "PRC-LAW-1",
        "text": "第一条 人工构造测试文本",
        "score": 1.0,
        "evidence_type": "statute",
        "source_url": "https://www.gov.cn/test",
        "effective_date": "2024-01-01",
        "expiry_date": None,
        "legal_status": "effective",
        "source_status": "human_verified_official",
    }
    values.update(overrides)
    return SearchHit(**values)


def test_temporal_filter_accepts_verified_effective_statute() -> None:
    accepted, rejected = filter_effective_statutes([statute()], "2025-01-01")
    assert len(accepted) == 1
    assert rejected == []


@pytest.mark.parametrize(
    ("overrides", "as_of", "reason"),
    [
        ({"source_status": "automatic_unreviewed"}, "2025-01-01", "source_not_human_verified"),
        ({"expiry_date": "2025-01-01"}, "2025-01-01", "expired"),
        ({"effective_date": "2026-01-01"}, "2025-01-01", "not_yet_effective"),
        (
            {"source_url": "https://www.gov.cn.example.com/fake"},
            "2025-01-01",
            "non_authoritative_source_url",
        ),
        ({}, None, "missing_as_of_date"),
    ],
)
def test_temporal_filter_rejects_unsafe_statute(overrides, as_of, reason) -> None:
    assert reason in assess_statute_hit(statute(**overrides), as_of)


def test_temporal_filter_rejects_invalid_date() -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        assess_statute_hit(statute(), "2025/01/01")
