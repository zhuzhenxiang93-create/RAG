"""Transparent lexical reranker for Lite mode."""

from typing import List

from app.retrieval.models import Candidate
from app.retrieval.tokenizer import tokenize


class LiteReranker:
    """Rerank using query coverage, section match and original fused score."""

    def rerank(self, query: str, candidates: List[Candidate]) -> List[Candidate]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return candidates

        for candidate in candidates:
            content_tokens = set(tokenize(candidate.chunk.content))
            coverage = len(query_tokens & content_tokens) / len(query_tokens)
            section_tokens = set(tokenize(candidate.chunk.section or ""))
            section_coverage = len(query_tokens & section_tokens) / len(query_tokens)
            exact_bonus = 1.0 if query.lower() in candidate.chunk.content.lower() else 0.0
            candidate.rerank_score = (
                0.65 * coverage + 0.2 * section_coverage + 0.15 * exact_bonus
            )
            candidate.score = 0.7 * candidate.rerank_score + 0.3 * min(
                (candidate.rrf_score or 0.0) * 60.0, 1.0
            )
            if "reranker" not in candidate.channels:
                candidate.channels.append("reranker")

        return sorted(
            candidates,
            key=lambda candidate: (-candidate.score, candidate.chunk.chunk_id),
        )
