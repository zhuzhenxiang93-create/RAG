"""Index lifecycle and adaptive retrieval orchestration."""

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from app.reranking.lite import LiteReranker
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import LiteDenseIndex
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import Candidate, IndexedChunk
from app.routing.query_router import QueryRouter
from app.schemas.documents import DocumentChunk
from app.schemas.search import (
    IndexBuildResponse,
    RetrievalPlan,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.storage.documents import DocumentRepository

if TYPE_CHECKING:
    from app.plugins.intent.classifier import IntentClassifier


class IndexService:
    """Build and query reproducible Lite indexes over persisted child chunks."""

    def __init__(
        self,
        data_dir: Path,
        intent_classifier: Optional["IntentClassifier"] = None,
    ) -> None:
        self.data_dir = data_dir
        self.repository = DocumentRepository(data_dir)
        self.bm25 = BM25Index()
        self.dense = LiteDenseIndex()
        self.router = QueryRouter()
        self.intent_classifier = intent_classifier
        self.reranker = LiteReranker()
        self.fingerprint = ""
        self.document_count = 0
        self.parent_count = 0
        self.child_count = 0
        self._lock = threading.RLock()

    def build(self) -> IndexBuildResponse:
        started = time.perf_counter()
        with self._lock:
            records = self.repository.list()
            indexed: List[IndexedChunk] = []
            parent_count = 0
            for record in records:
                detail = self.repository.get(record.document_id)
                if detail is None:
                    continue
                parents: Dict[str, DocumentChunk] = {
                    chunk.chunk_id: chunk for chunk in detail.chunks if chunk.level == "parent"
                }
                parent_count += len(parents)
                for child in (chunk for chunk in detail.chunks if chunk.level == "child"):
                    parent = parents.get(child.parent_id or "")
                    indexed.append(
                        IndexedChunk(
                            document_id=record.document_id,
                            filename=record.filename,
                            chunk_id=child.chunk_id,
                            parent_id=child.parent_id,
                            content=child.content,
                            parent_content=parent.content if parent else None,
                            page=child.page,
                            section=child.section,
                            content_type=child.content_type,
                        )
                    )
            self.bm25.build(indexed)
            self.dense.build(indexed)
            self.document_count = len(records)
            self.parent_count = parent_count
            self.child_count = len(indexed)
            self.fingerprint = self._fingerprint(records)
            self._write_manifest()
        return IndexBuildResponse(
            status="ready",
            fingerprint=self.fingerprint,
            document_count=self.document_count,
            child_chunk_count=self.child_count,
            parent_chunk_count=self.parent_count,
            build_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        timings: Dict[str, float] = {}
        plan = self.router.route(request.query)
        if self.intent_classifier is not None:
            started = time.perf_counter()
            try:
                intent_result = self.intent_classifier.predict(request.query, top_k=3)
                plan = self.router.apply_intent(plan, intent_result)
            except RuntimeError:
                # Intent routing is an optional optimization; retrieval remains available.
                pass
            timings["intent_router"] = self._elapsed(started)
        if self._current_fingerprint() != self.fingerprint:
            build_result = self.build()
            timings["build"] = build_result.build_ms

        if plan.query_type == "out_of_scope":
            return SearchResponse(
                query=request.query,
                strategy=request.strategy,
                plan=plan,
                index_fingerprint=self.fingerprint,
                total_candidates=0,
                results=[],
                timings_ms=timings,
            )

        candidate_limit = max(request.top_k * 8, 30)
        started = time.perf_counter()
        bm25_results = self.bm25.search(request.query, candidate_limit)
        timings["bm25"] = self._elapsed(started)

        started = time.perf_counter()
        dense_results = self.dense.search(request.query, candidate_limit)
        timings["dense"] = self._elapsed(started)

        candidates, effective_plan = self._select_strategy(
            request, plan, bm25_results, dense_results
        )
        total_candidates = len(candidates)
        if effective_plan.use_reranker and request.strategy in {
            "rrf_rerank",
            "adaptive",
        }:
            started = time.perf_counter()
            candidates = self.reranker.rerank(
                request.query, candidates[: max(request.top_k * 4, 20)]
            )
            timings["reranker"] = self._elapsed(started)

        candidates = self._deduplicate_parents(candidates)[: request.top_k]
        hits = [self._hit(candidate, rank) for rank, candidate in enumerate(candidates, 1)]
        return SearchResponse(
            query=request.query,
            strategy=request.strategy,
            plan=effective_plan,
            index_fingerprint=self.fingerprint,
            total_candidates=total_candidates,
            results=hits,
            timings_ms=timings,
        )

    def _select_strategy(
        self,
        request: SearchRequest,
        plan: RetrievalPlan,
        bm25_results: List[Tuple[IndexedChunk, float]],
        dense_results: List[Tuple[IndexedChunk, float]],
    ) -> Tuple[List[Candidate], RetrievalPlan]:
        if request.strategy == "bm25":
            selected = [
                Candidate(
                    chunk=chunk,
                    score=score,
                    bm25_score=score,
                    bm25_rank=rank,
                    channels=["bm25"],
                )
                for rank, (chunk, score) in enumerate(bm25_results, 1)
            ]
            return selected, plan.model_copy(update={"use_reranker": False})
        if request.strategy == "dense":
            selected = [
                Candidate(
                    chunk=chunk,
                    score=score,
                    dense_score=score,
                    dense_rank=rank,
                    channels=["dense"],
                )
                for rank, (chunk, score) in enumerate(dense_results, 1)
            ]
            return selected, plan.model_copy(update={"use_reranker": False})

        bm25_weight = plan.bm25_weight if request.strategy == "adaptive" else 1.0
        dense_weight = plan.dense_weight if request.strategy == "adaptive" else 1.0
        fused = reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
        if request.strategy == "rrf":
            return fused, plan.model_copy(update={"use_reranker": False})
        if request.strategy == "rrf_rerank":
            return fused, plan.model_copy(update={"use_reranker": True})
        return fused, plan

    @staticmethod
    def _deduplicate_parents(candidates: List[Candidate]) -> List[Candidate]:
        seen = set()
        result = []
        for candidate in candidates:
            key = candidate.chunk.parent_id or candidate.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)
        return result

    @staticmethod
    def _hit(candidate: Candidate, rank: int) -> SearchHit:
        chunk = candidate.chunk
        return SearchHit(
            document_id=chunk.document_id,
            filename=chunk.filename,
            chunk_id=chunk.chunk_id,
            parent_id=chunk.parent_id,
            page=chunk.page,
            section=chunk.section,
            content_type=chunk.content_type,
            content=chunk.content,
            parent_content=chunk.parent_content,
            score=round(candidate.score, 8),
            bm25_score=candidate.bm25_score,
            dense_score=candidate.dense_score,
            rrf_score=candidate.rrf_score,
            rerank_score=candidate.rerank_score,
            bm25_rank=candidate.bm25_rank,
            dense_rank=candidate.dense_rank,
            fused_rank=candidate.fused_rank,
            final_rank=rank,
            channels=candidate.channels,
        )

    def _current_fingerprint(self) -> str:
        return self._fingerprint(self.repository.list())

    @staticmethod
    def _fingerprint(records: List) -> str:
        payload = "|".join(
            "{}:{}:{}".format(record.document_id, record.sha256, record.child_chunk_count)
            for record in sorted(records, key=lambda item: item.document_id)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _write_manifest(self) -> None:
        index_dir = self.data_dir / "indexes"
        index_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "fingerprint": self.fingerprint,
            "document_count": self.document_count,
            "parent_chunk_count": self.parent_count,
            "child_chunk_count": self.child_count,
            "providers": {"sparse": "bm25-lite", "dense": "hashing-384"},
        }
        temporary = index_dir / "manifest.json.part"
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(index_dir / "manifest.json")

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000.0, 3)
