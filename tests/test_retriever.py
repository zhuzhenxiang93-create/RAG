from legalmind.retrieval.retriever import HybridRetriever
from legalmind.schemas import SearchHit


def make_hit(chunk_id: str, score: float) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        case_id=chunk_id,
        text=chunk_id,
        score=score,
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


def test_retriever_uses_reranker_before_case_deduplication():
    retriever = HybridRetriever(
        FakeIndex(),
        reranker=FakeReranker(),
        fusion_top_k=3,
        rerank_top_k=2,
    )
    results = retriever.search("query", candidate_k=3, final_k=1)
    assert results[0].chunk_id == "vector"
