from __future__ import annotations

from collections import defaultdict

from legalmind.schemas import SearchHit


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[SearchHit]],
    rrf_k: int = 60,
    predicted_labels: set[str] | None = None,
    label_boost: float = 0.15,
) -> list[SearchHit]:
    scores: dict[str, float] = defaultdict(float)
    hits: dict[str, SearchHit] = {}
    source_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for source, ranked in ranked_lists.items():
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.chunk_id] += 1.0 / (rrf_k + rank)
            hits[hit.chunk_id] = hit
            source_scores[hit.chunk_id][source] = hit.score
    predicted_labels = predicted_labels or set()
    for chunk_id, hit in hits.items():
        if predicted_labels.intersection(hit.accusations):
            scores[chunk_id] *= 1.0 + label_boost
        hit.score = scores[chunk_id]
        hit.source_scores = source_scores[chunk_id]
    return sorted(hits.values(), key=lambda item: item.score, reverse=True)


def deduplicate_cases(hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    selected: list[SearchHit] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.case_id in seen:
            continue
        selected.append(hit)
        seen.add(hit.case_id)
        if len(selected) >= top_k:
            break
    return selected
