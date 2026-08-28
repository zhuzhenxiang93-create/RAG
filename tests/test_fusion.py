from legalmind.retrieval.fusion import deduplicate_cases, reciprocal_rank_fusion
from legalmind.schemas import SearchHit


def hit(chunk_id: str, case_id: str, score: float, labels=None):
    return SearchHit(
        chunk_id=chunk_id,
        case_id=case_id,
        text=chunk_id,
        score=score,
        accusations=labels or [],
    )


def test_rrf_merges_sources_and_applies_soft_label_boost():
    fused = reciprocal_rank_fusion(
        {
            "bm25": [hit("a", "1", 9), hit("b", "2", 8, ["盗窃"])],
            "vector": [hit("a", "1", 0.9), hit("b", "2", 0.8, ["盗窃"])],
        },
        predicted_labels={"盗窃"},
        label_boost=0.5,
    )
    assert fused[0].chunk_id == "b"
    assert set(fused[1].source_scores) == {"bm25", "vector"}


def test_case_deduplication():
    values = [hit("a1", "a", 2), hit("a2", "a", 1), hit("b1", "b", 0.5)]
    assert [item.case_id for item in deduplicate_cases(values, 2)] == ["a", "b"]
