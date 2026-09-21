from __future__ import annotations

from legalmind.retrieval.fusion import deduplicate_cases, reciprocal_rank_fusion
from legalmind.retrieval.index import HybridIndex
from legalmind.retrieval.reranker import Reranker
from legalmind.schemas import LabelScore, SearchHit


class HybridRetriever:
    """BM25 + dense dual recall, RRF fusion, optional reranking, and case deduplication."""

    def __init__(
        self,
        index: HybridIndex,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        label_boost: float = 0.15,
        fusion_top_k: int = 50,
        rerank_top_k: int = 20,
        candidate_k: int = 100,
        final_k: int = 3,
    ):
        self.index = index
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.label_boost = label_boost
        self.fusion_top_k = fusion_top_k
        self.rerank_top_k = rerank_top_k
        self.candidate_k = candidate_k
        self.final_k = final_k

    @staticmethod
    def _label_aliases(labels: set[str] | None) -> set[str]:
        aliases: set[str] = set()
        for label in labels or set():
            value = label.strip()
            if not value:
                continue
            aliases.add(value)
            aliases.add(value.removesuffix("罪"))
        return aliases

    def search(
        self,
        query: str,
        predicted_labels: set[str] | None = None,
        candidate_k: int | None = None,
        final_k: int | None = None,
    ) -> list[SearchHit]:
        candidate_k = int(candidate_k or self.candidate_k)
        final_k = int(final_k or self.final_k)
        fused = reciprocal_rank_fusion(
            {
                "bm25": self.index.search_bm25(query, candidate_k),
                "vector": self.index.search_vector(query, candidate_k),
            },
            rrf_k=self.rrf_k,
            predicted_labels=self._label_aliases(predicted_labels),
            label_boost=self.label_boost,
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
        """Pipeline-compatible entry point using classifier labels as a soft retrieval prior."""
        predicted_labels = {item.label for item in accusations}
        return self.search(
            query,
            predicted_labels=predicted_labels,
            candidate_k=self.candidate_k,
            final_k=top_k,
        )
