from __future__ import annotations

from legalmind.retrieval.fusion import deduplicate_cases, reciprocal_rank_fusion
from legalmind.retrieval.index import HybridIndex
from legalmind.retrieval.reranker import Reranker
from legalmind.schemas import LabelScore, SearchHit


class HybridRetriever:
    """Hard accusation-gated BM25 + dense recall, RRF fusion, reranking, and case deduplication."""

    def __init__(
        self,
        index: HybridIndex,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        fusion_top_k: int = 50,
        rerank_top_k: int = 20,
        candidate_k: int = 100,
        final_k: int = 3,
    ):
        self.index = index
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.fusion_top_k = fusion_top_k
        self.rerank_top_k = rerank_top_k
        self.candidate_k = candidate_k
        self.final_k = final_k

    @staticmethod
    def _canonical_labels(labels: set[str] | None) -> set[str]:
        return {
            value.strip().removesuffix("罪")
            for value in labels or set()
            if value.strip()
        }

    def search(
        self,
        query: str,
        predicted_labels: set[str] | None = None,
        candidate_k: int | None = None,
        final_k: int | None = None,
    ) -> list[SearchHit]:
        candidate_k = int(candidate_k or self.candidate_k)
        final_k = int(final_k or self.final_k)
        labels = self._canonical_labels(predicted_labels)

        if labels:
            bm25 = self.index.search_bm25_filtered(query, labels, candidate_k)
            dense = self.index.search_vector_filtered(query, labels, candidate_k)
        else:
            bm25 = self.index.search_bm25(query, candidate_k)
            dense = self.index.search_vector(query, candidate_k)

        fused = reciprocal_rank_fusion(
            {"bm25": bm25, "vector": dense},
            rrf_k=self.rrf_k,
        )
        candidates = fused[: self.fusion_top_k]
        if self.reranker:
            candidates = self.reranker.rerank(query, candidates, self.rerank_top_k)
        return deduplicate_cases(candidates, final_k)

    def search_by_accusations(
        self,
        query: str,
        accusations: list[LabelScore],
        top_k: int = 3,
    ) -> list[SearchHit]:
        """Search only cases whose accusation is in the classifier-selected set."""
        selected_labels = {item.label for item in accusations}
        return self.search(
            query,
            predicted_labels=selected_labels,
            candidate_k=self.candidate_k,
            final_k=top_k,
        )
