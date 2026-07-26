"""Auditable confidence calculation."""

from typing import List, Tuple

from app.schemas.qa import ConfidenceBreakdown, ConflictItem, EvidenceItem
from app.schemas.search import SearchResponse


class ConfidenceCalculator:
    """Combine retrieval and evidence signals without asking an LLM for probability."""

    def calculate(
        self,
        retrieval: SearchResponse,
        evidence: List[EvidenceItem],
        conflicts: List[ConflictItem],
    ) -> Tuple[float, ConfidenceBreakdown]:
        supporting = [item for item in evidence if item.relation == "support"]
        relevant = [item for item in evidence if item.relation != "irrelevant"]

        evidence_relevance = (
            sum(item.relevance for item in supporting[:3]) / min(len(supporting), 3)
            if supporting
            else 0.0
        )
        top_hits = retrieval.results[:3]
        channel_agreement = (
            sum(
                "bm25" in hit.channels and "dense" in hit.channels
                for hit in top_hits
            )
            / len(top_hits)
            if top_hits
            else 0.0
        )
        ranking_margin = self._ranking_margin(retrieval)
        citation_coverage = min(len(supporting) / 2.0, 1.0)
        conflict_penalty = min(len(conflicts) * 0.5, 1.0)

        score = (
            0.45 * evidence_relevance
            + 0.2 * channel_agreement
            + 0.15 * ranking_margin
            + 0.2 * citation_coverage
            - 0.5 * conflict_penalty
        )
        if not relevant:
            score = 0.0
        score = round(max(0.0, min(score, 1.0)), 4)
        breakdown = ConfidenceBreakdown(
            evidence_relevance=round(evidence_relevance, 4),
            channel_agreement=round(channel_agreement, 4),
            ranking_margin=round(ranking_margin, 4),
            citation_coverage=round(citation_coverage, 4),
            conflict_penalty=round(conflict_penalty, 4),
        )
        return score, breakdown

    @staticmethod
    def _ranking_margin(retrieval: SearchResponse) -> float:
        if not retrieval.results:
            return 0.0
        if len(retrieval.results) == 1:
            return 1.0
        first = retrieval.results[0].score
        second = retrieval.results[1].score
        denominator = max(abs(first), 1e-8)
        return max(0.0, min((first - second) / denominator, 1.0))
