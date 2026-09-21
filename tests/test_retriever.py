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


class FakeFilteredIndex:
    def __init__(self):
        self.bm25_labels = None
        self.vector_labels = None

    def search_bm25_filtered(self, query, labels, top_k):
        self.bm25_labels = labels
        return [
            make_hit("theft-bm25", 10.0, ["盗窃"]),
            make_hit("theft-shared", 9.0, ["盗窃"]),
        ][:top_k]

    def search_vector_filtered(self, query, labels, top_k):
        self.vector_labels = labels
        return [
            make_hit("theft-shared", 0.9, ["盗窃"]),
            make_hit("theft-dense", 0.8, ["盗窃"]),
        ][:top_k]


def test_retriever_uses_reranker_before_case_deduplication():
    retriever = HybridRetriever(
        FakeIndex(),
        reranker=FakeReranker(),
        fusion_top_k=3,
        rerank_top_k=2,
    )
    results = retriever.search("query", candidate_k=3, final_k=1)
    assert results[0].chunk_id == "vector"


def test_pipeline_entry_hard_filters_to_predicted_charge_scope():
    index = FakeFilteredIndex()
    retriever = HybridRetriever(
        index,
        reranker=None,
        fusion_top_k=3,
        candidate_k=3,
        final_k=3,
    )
    results = retriever.search_by_accusations(
        "秘密取得他人财物",
        [LabelScore(label_id=1, label="盗窃罪", probability=0.8)],
        top_k=3,
    )
    assert index.bm25_labels == {"盗窃"}
    assert index.vector_labels == {"盗窃"}
    assert results
    assert all("盗窃" in hit.accusations for hit in results)
