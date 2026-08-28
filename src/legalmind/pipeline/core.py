from __future__ import annotations

import time
from typing import Any

from legalmind.generation.structured import deterministic_grounded_analysis, validate_citations


class LegalMindPipeline:
    def __init__(
        self,
        classifier: Any | None = None,
        retriever: Any | None = None,
        statute_retriever: Any | None = None,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.statute_retriever = statute_retriever

    def _retrieve_statutes(self, fact: str, labels: list[str], top_k: int) -> list[Any]:
        if self.statute_retriever is None:
            return []
        selected = []
        seen_articles: set[int] = set()
        # A short accusation query retrieves the offense-defining clause more
        # reliably than a long fact dominated by generic procedural language.
        for label in labels:
            for hit in self.statute_retriever.search(label, top_k=1):
                articles = set(hit.relevant_articles)
                if articles and not articles.intersection(seen_articles):
                    selected.append(hit)
                    seen_articles.update(articles)
                    break
            if len(selected) >= top_k:
                return selected
        # Add at most one fact-only clause for sentencing circumstances such as
        # surrender. More low-ranked fact matches are usually procedural noise.
        for hit in self.statute_retriever.search(fact, top_k=top_k):
            articles = set(hit.relevant_articles)
            if articles and not articles.intersection(seen_articles):
                selected.append(hit)
                break
        return selected

    def analyze(self, fact: str, top_k: int = 5) -> dict:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        labels = []
        classification = {"status": "degraded_no_classifier", "labels": []}
        if self.classifier is not None:
            phase = time.perf_counter()
            result = self.classifier.predict(fact)
            status = "uncertain_below_threshold" if result.used_fallback else "ok"
            classification = {"status": status, **result.model_dump()}
            labels = [row.label for row in result.labels]
            if result.used_fallback:
                labels = labels[:3]
            timings["classification_seconds"] = time.perf_counter() - phase

        hits = []
        retrieval_status = "degraded_no_index"
        if self.retriever is not None:
            phase = time.perf_counter()
            if hasattr(self.retriever, "search"):
                hits = self.retriever.search(fact, top_k=top_k)
            retrieval_status = "ok"
            timings["retrieval_seconds"] = time.perf_counter() - phase

        statute_hits = []
        statute_status = "degraded_no_statute_index"
        if self.statute_retriever is not None:
            phase = time.perf_counter()
            statute_hits = self._retrieve_statutes(fact, labels, top_k)
            statute_status = "ok"
            timings["statute_retrieval_seconds"] = time.perf_counter() - phase
        retrieved_articles = {article for hit in statute_hits for article in hit.relevant_articles}
        analysis = deterministic_grounded_analysis(fact, labels, hits, sorted(retrieved_articles))
        citation_check = validate_citations(
            analysis,
            {hit.case_id for hit in hits},
            retrieved_articles,
        )
        timings["total_seconds"] = time.perf_counter() - started
        return {
            "classification": classification,
            "retrieval": {
                "status": retrieval_status,
                "evidence": [hit.model_dump() for hit in hits],
                "statute_status": statute_status,
                "statutes": [hit.model_dump() for hit in statute_hits],
            },
            "analysis": analysis.model_dump(),
            "citation_validation": citation_check,
            "timings": timings,
            "disclaimer": "仅用于算法实验，不构成法律意见。",
        }
