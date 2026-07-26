"""Reproducible built-in benchmark runner."""

import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from app.evaluation.metrics import (
    binary_precision_recall,
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.indexing.service import IndexService
from app.schemas.evaluation import EvaluationRunSummary
from app.schemas.qa import QARequest
from app.schemas.search import SearchRequest
from app.services.documents import DocumentService
from app.services.qa import QAService


class EvaluationRunner:
    """Run retrieval ablations and trustworthy-QA checks in isolated storage."""

    STRATEGIES = ("bm25", "dense", "rrf", "rrf_rerank", "adaptive")
    TOP_K = (1, 3, 5)

    def __init__(self, project_root: Path, artifact_root: Path) -> None:
        self.project_root = project_root
        self.artifact_root = artifact_root

    def run(self, benchmark: str = "lite_v1") -> EvaluationRunSummary:
        benchmark_dir = self.project_root / "data" / "eval" / benchmark
        questions = json.loads(
            (benchmark_dir / "questions.json").read_text(encoding="utf-8")
        )
        run_id = "{}-{}".format(
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            uuid4().hex[:8],
        )

        with tempfile.TemporaryDirectory(prefix="docmind-eval-") as temporary:
            data_dir = Path(temporary)
            document_service = DocumentService(data_dir, max_upload_mb=10)
            for path in sorted((benchmark_dir / "corpus").glob("*")):
                if path.is_file():
                    with path.open("rb") as stream:
                        document_service.ingest(path.name, stream)
            index_service = IndexService(data_dir)
            build = index_service.build()
            retrieval_metrics, retrieval_raw = self._retrieval(
                index_service, questions
            )
            qa_metrics, qa_raw = self._qa(index_service, questions)

        payload = {
            "run_id": run_id,
            "status": "completed",
            "benchmark": benchmark,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "warning": (
                "Small manually authored Lite benchmark. "
                "Do not present these values as production or model-quality metrics."
            ),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "app_mode": "lite",
                "sparse_provider": "bm25-lite",
                "dense_provider": "hashing-384",
                "reranker": "lexical-lite",
            },
            "configuration": {
                "strategies": list(self.STRATEGIES),
                "top_k": list(self.TOP_K),
                "question_count": len(questions),
                "answerable_count": sum(item["answerable"] for item in questions),
                "corpus_document_count": build.document_count,
                "index_fingerprint": build.fingerprint,
            },
            "retrieval_metrics": retrieval_metrics,
            "qa_metrics": qa_metrics,
            "raw_predictions": {
                "retrieval": retrieval_raw,
                "qa": qa_raw,
            },
        }
        run_dir = self.artifact_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        artifact = run_dir / "results.json"
        artifact.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return EvaluationRunSummary(
            run_id=run_id,
            status="completed",
            benchmark=benchmark,
            question_count=len(questions),
            answerable_count=sum(item["answerable"] for item in questions),
            retrieval_metrics=retrieval_metrics,
            qa_metrics=qa_metrics,
            artifact_path=str(artifact),
        )

    def load(self, run_id: str) -> Dict:
        if not run_id or any(character not in "0123456789TZ-abcdef" for character in run_id):
            raise ValueError("Invalid evaluation run ID")
        path = self.artifact_root / run_id / "results.json"
        if not path.is_file():
            raise FileNotFoundError(run_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def _retrieval(self, index_service: IndexService, questions: List[Dict]):
        answerable = [item for item in questions if item["answerable"]]
        raw: Dict[str, List[Dict]] = {}
        aggregate: Dict[str, Dict] = {}
        for strategy in self.STRATEGIES:
            rows: List[Dict] = []
            latencies: List[float] = []
            for question in answerable:
                started = time.perf_counter()
                response = index_service.search(
                    SearchRequest(
                        query=question["query"],
                        strategy=strategy,
                        top_k=max(self.TOP_K),
                    )
                )
                elapsed = (time.perf_counter() - started) * 1000.0
                latencies.append(elapsed)
                retrieved = self._unique([hit.filename for hit in response.results])
                rows.append(
                    {
                        "id": question["id"],
                        "query": question["query"],
                        "gold_documents": question["gold_documents"],
                        "retrieved_documents": retrieved,
                        "router_type": response.plan.query_type,
                        "latency_ms": round(elapsed, 4),
                    }
                )
            metrics = {
                "mrr": round(mean(
                    reciprocal_rank(row["retrieved_documents"], row["gold_documents"])
                    for row in rows
                ), 4),
                "latency_ms": {
                    "mean": round(mean(latencies), 4),
                    "p50": round(percentile(latencies, 0.5), 4),
                    "p95": round(percentile(latencies, 0.95), 4),
                },
            }
            for k in self.TOP_K:
                metrics["precision@{}".format(k)] = round(mean(
                    precision_at_k(row["retrieved_documents"], row["gold_documents"], k)
                    for row in rows
                ), 4)
                metrics["recall@{}".format(k)] = round(mean(
                    recall_at_k(row["retrieved_documents"], row["gold_documents"], k)
                    for row in rows
                ), 4)
                metrics["ndcg@{}".format(k)] = round(mean(
                    ndcg_at_k(row["retrieved_documents"], row["gold_documents"], k)
                    for row in rows
                ), 4)
            aggregate[strategy] = metrics
            raw[strategy] = rows
        return aggregate, raw

    def _qa(self, index_service: IndexService, questions: List[Dict]):
        service = QAService(index_service)
        rows = []
        for question in questions:
            response = service.answer(
                QARequest(query=question["query"], strategy="adaptive", top_k=5)
            )
            cited_documents = self._unique(
                [citation.filename for citation in response.citations]
            )
            rows.append(
                {
                    "id": question["id"],
                    "expected_decision": question["expected_decision"],
                    "predicted_decision": response.decision,
                    "confidence": response.confidence,
                    "gold_documents": question["gold_documents"],
                    "cited_documents": cited_documents,
                    "conflict_count": len(response.conflicts),
                    "answer": response.answer,
                }
            )
        expected_abstain = [
            row["expected_decision"] == "abstain" for row in rows
        ]
        predicted_abstain = [
            row["predicted_decision"] == "abstain" for row in rows
        ]
        abstention = binary_precision_recall(expected_abstain, predicted_abstain)
        citation_rows = [row for row in rows if row["gold_documents"]]
        expected_conflict = [
            row["expected_decision"] == "clarify" for row in rows
        ]
        predicted_conflict = [
            row["predicted_decision"] == "clarify" and row["conflict_count"] > 0
            for row in rows
        ]
        conflict = binary_precision_recall(expected_conflict, predicted_conflict)
        metrics = {
            "decision_accuracy": round(mean(
                row["expected_decision"] == row["predicted_decision"] for row in rows
            ), 4),
            "abstention_precision": round(abstention["precision"], 4),
            "abstention_recall": round(abstention["recall"], 4),
            "conflict_precision": round(conflict["precision"], 4),
            "conflict_recall": round(conflict["recall"], 4),
            "citation_precision": round(mean(
                precision_at_k(
                    row["cited_documents"],
                    row["gold_documents"],
                    max(len(row["cited_documents"]), 1),
                )
                for row in citation_rows
            ), 4),
            "citation_recall": round(mean(
                recall_at_k(
                    row["cited_documents"],
                    row["gold_documents"],
                    max(len(row["cited_documents"]), 1),
                )
                for row in citation_rows
            ), 4),
        }
        return metrics, rows

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(values))
