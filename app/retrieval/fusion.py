"""Rank-based fusion."""

from typing import Dict, Iterable, List, Tuple

from app.retrieval.models import Candidate, IndexedChunk


def reciprocal_rank_fusion(
    bm25_results: Iterable[Tuple[IndexedChunk, float]],
    dense_results: Iterable[Tuple[IndexedChunk, float]],
    *,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    rrf_k: int = 60,
) -> List[Candidate]:
    """Fuse only ranks in which a candidate actually appears."""
    candidates: Dict[str, Candidate] = {}

    for rank, (chunk, raw_score) in enumerate(bm25_results, 1):
        candidate = candidates.setdefault(chunk.chunk_id, Candidate(chunk=chunk))
        candidate.bm25_score = raw_score
        candidate.bm25_rank = rank
        candidate.channels.append("bm25")
        candidate.rrf_score = (candidate.rrf_score or 0.0) + bm25_weight / (rrf_k + rank)

    for rank, (chunk, raw_score) in enumerate(dense_results, 1):
        candidate = candidates.setdefault(chunk.chunk_id, Candidate(chunk=chunk))
        candidate.dense_score = raw_score
        candidate.dense_rank = rank
        candidate.channels.append("dense")
        candidate.rrf_score = (candidate.rrf_score or 0.0) + dense_weight / (rrf_k + rank)

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (
            -(candidate.rrf_score or 0.0),
            candidate.chunk.chunk_id,
        ),
    )
    for rank, candidate in enumerate(ranked, 1):
        candidate.fused_rank = rank
        candidate.score = candidate.rrf_score or 0.0
    return ranked
