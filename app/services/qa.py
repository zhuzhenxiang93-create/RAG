"""Trustworthy QA orchestration."""

from typing import Tuple

from app.evidence.analyzer import EvidenceAnalyzer
from app.evidence.confidence import ConfidenceCalculator
from app.generation.extractive import ExtractiveGenerator
from app.indexing.service import IndexService
from app.schemas.qa import AnswerDecision, QARequest, QAResponse
from app.schemas.search import SearchRequest


class QAService:
    """Retrieve, verify and selectively answer."""

    def __init__(self, index_service: IndexService) -> None:
        self.index_service = index_service
        self.analyzer = EvidenceAnalyzer()
        self.confidence = ConfidenceCalculator()
        self.generator = ExtractiveGenerator()

    def answer(self, request: QARequest) -> QAResponse:
        retrieval = self.index_service.search(
            SearchRequest(
                query=request.query,
                strategy=request.strategy,
                top_k=request.top_k,
            )
        )
        evidence, conflicts = self.analyzer.analyze(request.query, retrieval)
        confidence, breakdown = self.confidence.calculate(
            retrieval, evidence, conflicts
        )
        decision, reason = self._decision(
            retrieval.plan.query_type,
            confidence,
            sum(item.relation == "support" for item in evidence),
            bool(conflicts),
        )
        answer = self.generator.generate(request.query, evidence, conflicts)
        citations = [item for item in evidence if item.relation != "irrelevant"]
        return QAResponse(
            answer=answer,
            decision=decision,
            confidence=confidence,
            confidence_breakdown=breakdown,
            citations=citations,
            conflicts=conflicts,
            reason=reason,
            retrieval=retrieval,
        )

    @staticmethod
    def _decision(
        query_type: str,
        confidence: float,
        support_count: int,
        has_conflict: bool,
    ) -> Tuple[AnswerDecision, str]:
        if query_type == "out_of_scope":
            return "abstain", "The query is outside the indexed-document scope."
        if has_conflict:
            return "clarify", "Conflicting documents require a version or time clarification."
        if support_count == 0 or confidence < 0.3:
            return "abstain", "No sufficiently relevant evidence was found."
        if confidence < 0.52 or support_count < 1:
            return "clarify", "Evidence exists but confidence is below the answer threshold."
        return "answer", "Evidence is relevant, cited and free of detected conflicts."
