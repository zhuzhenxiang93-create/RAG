from __future__ import annotations

from legalmind.pipeline.core import LegalMindPipeline
from legalmind.schemas import SearchHit


class FakeRetriever:
    def __init__(self, hits: list[SearchHit]):
        self.hits = hits

    def search(self, _query: str, top_k: int = 5) -> list[SearchHit]:
        return self.hits[:top_k]


def test_pipeline_keeps_case_and_statute_evidence_separate() -> None:
    case = SearchHit(
        chunk_id="CASE-1#0",
        case_id="CASE-1",
        text="持刀伤人并造成轻伤",
        score=1.0,
        accusations=["故意伤害"],
    )
    statute = SearchHit(
        chunk_id="PRC-CRIMINAL-LAW-234#article",
        case_id="PRC-CRIMINAL-LAW-234",
        text="刑法第二百三十四条",
        score=1.0,
        relevant_articles=[234],
    )
    result = LegalMindPipeline(
        retriever=FakeRetriever([case]),
        statute_retriever=FakeRetriever([statute]),
    ).analyze("被告人持刀将他人刺成轻伤。")
    assert result["retrieval"]["evidence"][0]["case_id"] == "CASE-1"
    assert result["retrieval"]["statutes"][0]["relevant_articles"] == [234]
    assert result["analysis"]["relevant_articles"] == [234]
    assert result["citation_validation"]["valid"] is True


def test_pipeline_has_explicit_cpu_degradation_status() -> None:
    result = LegalMindPipeline().analyze("仅用于测试的匿名案件事实。")
    assert result["classification"]["status"] == "degraded_no_classifier"
    assert result["retrieval"]["status"] == "degraded_no_index"
    assert result["retrieval"]["statute_status"] == "degraded_no_statute_index"
    assert result["analysis"]["requires_manual_review"] is True
