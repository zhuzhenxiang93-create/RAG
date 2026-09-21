from legalmind.retrieval.retriever import HybridRetriever
from legalmind.schemas import LabelScore, SearchHit


def make_hit(
    chunk_id: str,
    score: float,
    accusations: list[str] | None = None,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        case_id=chunk_id,
        text=chunk_id,
        score=score,
        accusations=accusations or [],
    )


class FakeIndex:
    def search_bm25(self, query, top_k):
        return [make_hit("bm25", 10.0), make_hit("shared", 9.0)]

    def search_vector(self, query, top_k):
        return [make_hit("shared", 0.9), make_hit("vector", 0.8)]


class FakeReranker:
    def rerank(self, query, hits, top_k):
        for hit in hits:
            hit.score = 1.0 if hit.chunk_id == "vector" else 0.0
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]


class FakeLabelIndex:
    def search_bm25(self, query, top_k):
        return [
            make_hit("fraud", 10.0, ["诈骗"]),
            make_hit("theft", 9.0, ["盗窃"]),
        ]

    def search_vector(self, query, top_k):
        return [
            make_hit("fraud", 0.9, ["诈骗"]),
            make_hit("theft", 0.8, ["盗窃"]),
        ]


def test_retriever_uses_reranker_before_case_deduplication():
    retriever = HybridRetriever(
        FakeIndex(),
        reranker=FakeReranker(),
        fusion_top_k=3,
        rerank_top_k=2,
    )
    results = retriever.search("query", candidate_k=3, final_k=1)
    assert results[0].chunk_id == "vector"


def test_pipeline_entry_uses_predicted_charge_as_soft_rrf_boost():
    retriever = HybridRetriever(
        FakeLabelIndex(),
        reranker=None,
        label_boost=0.15,
        fusion_top_k=2,
        candidate_k=2,
        final_k=1,
    )
    results = retriever.search_by_accusations(
        "秘密取得他人财物",
        [LabelScore(label_id=1, label="盗窃罪", probability=0.8)],
        top_k=1,
    )
    assert results[0].case_id == "theft"
