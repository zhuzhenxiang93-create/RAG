from __future__ import annotations

import time
from typing import Any

from legalmind.generation.evidence_packet import build_evidence_packet
from legalmind.generation.structured import (
    deterministic_grounded_analysis,
    validate_citations,
)
from legalmind.pipeline.contracts import LegalCaseAnalysisResponse
from legalmind.pipeline.evidence_firewall import evidence_firewall, safe_evidence_payload
from legalmind.retrieval.temporal import filter_effective_statutes
from legalmind.schemas import LabelScore
from legalmind.sentencing.schemas import PredictedAccusation, SentencingResponse


def _canonical_accusation(value: str) -> str:
    return value.strip().removesuffix("罪")


def select_retrieval_label_scores(
    scores: list[LabelScore], used_fallback: bool, max_fallback_candidates: int = 3
) -> list[LabelScore]:
    """Keep calibrated labels, but broaden uncertain fallback retrieval to Top-K."""
    if not used_fallback:
        return list(scores)
    return list(scores[: max(1, max_fallback_candidates)])


class LegalMindPipeline:
    def __init__(
        self,
        classifier: Any | None = None,
        retriever: Any | None = None,
        statute_retriever: Any | None = None,
        sentencing_service: Any | None = None,
        grounded_analysis_service: Any | None = None,
        initialization_warnings: list[str] | None = None,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.statute_retriever = statute_retriever
        self.sentencing_service = sentencing_service
        self.grounded_analysis_service = grounded_analysis_service
        self.initialization_warnings = initialization_warnings or []

    def _retrieve_statutes(
        self, fact: str, labels: list[str], top_k: int, as_of_date: str | None
    ) -> tuple[list[Any], list[dict[str, object]]]:
        if self.statute_retriever is None:
            return [], []
        selected = []
        rejected: list[dict[str, object]] = []
        seen_articles: set[int] = set()
        # A short accusation query retrieves the offense-defining clause more
        # reliably than a long fact dominated by generic procedural language.
        for label in labels:
            candidates = self.statute_retriever.search(label, top_k=max(5, top_k))
            accepted, rejected_candidates = filter_effective_statutes(candidates, as_of_date)
            rejected.extend(rejected_candidates)
            for hit in accepted:
                articles = set(hit.relevant_articles)
                if articles and not articles.intersection(seen_articles):
                    selected.append(hit)
                    seen_articles.update(articles)
                    break
            if len(selected) >= top_k:
                return selected, rejected
        # Add at most one fact-only clause for sentencing circumstances such as
        # surrender. More low-ranked fact matches are usually procedural noise.
        candidates = self.statute_retriever.search(fact, top_k=max(top_k * 3, top_k))
        accepted, rejected_candidates = filter_effective_statutes(candidates, as_of_date)
        rejected.extend(rejected_candidates)
        for hit in accepted:
            articles = set(hit.relevant_articles)
            if articles and not articles.intersection(seen_articles):
                selected.append(hit)
                break
        return selected, rejected

    def analyze(
        self,
        fact: str,
        accusations: list[str] | None = None,
        top_k: int = 3,
        as_of_date: str | None = None,
    ) -> LegalCaseAnalysisResponse:
        top_k = min(max(top_k, 1), 3)
        timings: dict[str, float] = {}
        started = time.perf_counter()
        labels: list[str] = []
        label_scores: list[LabelScore] = []
        classification = {"status": "degraded_no_classifier", "labels": []}
        if accusations:
            labels = list(dict.fromkeys(value.strip() for value in accusations if value.strip()))
            label_scores = [
                LabelScore(label_id=index, label=label, probability=1.0)
                for index, label in enumerate(labels)
            ]
            classification = {
                "status": "provided",
                "labels": [item.model_dump() for item in label_scores],
                "thresholds": {},
                "used_fallback": False,
                "max_probability": 1.0,
            }
        elif self.classifier is not None:
            phase = time.perf_counter()
            result = self.classifier.predict(fact)
            status = "uncertain_below_threshold" if result.used_fallback else "ok"
            classification = {"status": status, **result.model_dump()}
            label_scores = select_retrieval_label_scores(result.labels, result.used_fallback)
            labels = [row.label for row in label_scores]
            classification["labels"] = [item.model_dump() for item in label_scores]
            timings["classification_seconds"] = time.perf_counter() - phase

        hits = []
        retrieval_status = "degraded_no_index"
        if self.retriever is not None:
            phase = time.perf_counter()
            if label_scores and hasattr(self.retriever, "search_by_accusations"):
                hits = self.retriever.search_by_accusations(fact, label_scores, top_k=top_k)
            elif hasattr(self.retriever, "search"):
                candidates = self.retriever.search(fact, top_k=max(top_k * 20, top_k))
                requested = {_canonical_accusation(label) for label in labels}
                hits = [
                    hit
                    for hit in candidates
                    if not requested
                    or requested.intersection(
                        _canonical_accusation(value) for value in hit.accusations
                    )
                ][:top_k]
            retrieval_status = "ok" if hits else "no_match"
            timings["retrieval_seconds"] = time.perf_counter() - phase

        statute_hits = []
        rejected_statutes: list[dict[str, object]] = []
        statute_status = "degraded_no_statute_index"
        if self.statute_retriever is not None:
            phase = time.perf_counter()
            statute_hits, rejected_statutes = self._retrieve_statutes(
                fact, labels, top_k, as_of_date
            )
            statute_status = "ok" if statute_hits else "no_verified_applicable_statute"
            timings["statute_retrieval_seconds"] = time.perf_counter() - phase
        retrieved_articles = {article for hit in statute_hits for article in hit.relevant_articles}
        analysis = deterministic_grounded_analysis(fact, labels, hits, sorted(retrieved_articles))
        citation_check = validate_citations(
            analysis,
            {hit.case_id for hit in hits},
            retrieved_articles,
        )
        firewall = evidence_firewall(
            citation_check, hits, statute_hits, rejected_statutes, as_of_date
        )
        sentencing = SentencingResponse(
            status="degraded_no_sentencing_model",
            warnings=["未加载结构化量刑模型。"],
        )
        if self.sentencing_service is not None:
            phase = time.perf_counter()
            sentencing = self.sentencing_service.predict(
                {
                    "fact": fact,
                    "predicted_accusations": [
                        PredictedAccusation(name=item.label, probability=item.probability)
                        for item in label_scores
                    ],
                    "top_k": top_k,
                }
            )
            timings["sentencing_seconds"] = time.perf_counter() - phase
        grounded_generation: dict[str, Any] = {
            "status": "disabled",
            "analysis": None,
            "validation": {},
        }
        if self.grounded_analysis_service is not None:
            phase = time.perf_counter()
            packet = build_evidence_packet(
                fact,
                labels,
                hits,
                statute_hits,
                as_of_date=as_of_date,
                sentencing_baseline=sentencing.model_dump(mode="json"),
            )
            grounded_analysis, generation_report = self.grounded_analysis_service.analyze(packet)
            fallback_used = bool(generation_report.get("fallback_used"))
            grounded_generation = {
                "status": "fallback" if fallback_used else "ok",
                "analysis": grounded_analysis.model_dump(mode="json"),
                "validation": generation_report,
            }
            timings["grounded_generation_seconds"] = time.perf_counter() - phase
        timings["total_seconds"] = time.perf_counter() - started
        requires_manual_review = (
            classification["status"] not in {"ok", "provided"}
            or analysis.requires_manual_review
            or sentencing.requires_manual_review
            or firewall["requires_manual_review"]
            or not citation_check["valid"]
            or grounded_generation["status"] == "fallback"
            or (
                grounded_generation["analysis"] is not None
                and grounded_generation["analysis"]["requires_manual_review"]
            )
        )
        return LegalCaseAnalysisResponse.model_validate(
            {
                "schema_version": "legal-case-analysis-v1",
                "classification": classification,
                "retrieval": {
                    "status": retrieval_status,
                    "evidence": [safe_evidence_payload(hit) for hit in hits],
                    "statute_status": statute_status,
                    "statutes": [safe_evidence_payload(hit) for hit in statute_hits],
                },
                "analysis": analysis.model_dump(),
                "citation_validation": citation_check,
                "evidence_firewall": firewall,
                "sentencing": sentencing,
                "grounded_generation": grounded_generation,
                "legal_as_of_date": as_of_date,
                "initialization_warnings": self.initialization_warnings,
                "timings": timings,
                "requires_manual_review": requires_manual_review,
                "disclaimer": (
                    "模型基于历史案件；法规结论仅基于指定日期且经核验的"
                    "权威来源。"
                    "仅用于算法实验，不构成法律意见。"
                ),
            }
        )
