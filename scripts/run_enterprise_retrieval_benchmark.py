"""Measure how source routing changes EnterpriseRAG-Bench retrieval quality."""

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from app.plugins.enterprise_router.data import read_jsonl
from app.plugins.enterprise_router.schema import SOURCE_TYPES
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import LiteDenseIndex
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import IndexedChunk


DOCUMENT_ID = re.compile(r"(dsid_[0-9a-fA-F]+)")


class SentenceTransformerIndex:
    """Optional real embedding baseline; the hashing backend is smoke-test only."""

    def __init__(self, model_name: str, batch_size: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "sentence-transformers is required for --dense-backend sentence-transformers"
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.chunks: List[IndexedChunk] = []
        self.matrix = np.empty((0, 0), dtype=np.float32)

    def build(self, chunks: List[IndexedChunk]) -> None:
        self.chunks = list(chunks)
        self.matrix = np.asarray(
            self.model.encode(
                [chunk.content for chunk in chunks],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )

    def search(self, query: str, limit: int):
        if not self.chunks:
            return []
        vector = np.asarray(
            self.model.encode([query], normalize_embeddings=True)[0],
            dtype=np.float32,
        )
        scores = self.matrix @ vector
        indices = np.argsort(-scores, kind="stable")[:limit]
        return [
            (self.chunks[int(index)], float(scores[int(index)]))
            for index in indices
        ]


def source_from_path(path: Path, corpus_dir: Path) -> str:
    relative = path.relative_to(corpus_dir)
    candidates = [part.lower().replace("-", "_") for part in relative.parts[:-1]]
    candidates.extend(part.lower().replace("-", "_") for part in path.stem.split("_"))
    for source in SOURCE_TYPES:
        if source in candidates:
            return source
    raise ValueError(
        "Cannot infer source for {}. Extract files under corpus/<source_type>/.".format(path)
    )


def load_corpus(corpus_dir: Path, limit: Optional[int]) -> Dict[str, List[IndexedChunk]]:
    grouped: Dict[str, List[IndexedChunk]] = {source: [] for source in SOURCE_TYPES}
    paths = sorted(corpus_dir.rglob("*.txt"))
    if limit:
        paths = paths[:limit]
    for path in paths:
        match = DOCUMENT_ID.search(path.name)
        if not match:
            continue
        source = source_from_path(path, corpus_dir)
        document_id = match.group(1)
        content = path.read_text(encoding="utf-8", errors="replace")
        grouped[source].append(
            IndexedChunk(
                document_id=document_id,
                filename=path.name,
                chunk_id=document_id,
                content=content,
                content_type="text/plain",
            )
        )
    if not any(grouped.values()):
        raise ValueError("No dsid_*.txt documents found under {}".format(corpus_dir))
    return grouped


def load_routes(path: Optional[Path]) -> Dict[str, List[str]]:
    if not path:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    routes = {}
    for row in payload.get("predictions", []):
        prediction = row.get("prediction")
        if isinstance(prediction, str):
            prediction = json.loads(prediction)
        if prediction:
            routes[str(row["id"])] = list(prediction["sources"])
    return routes


def unique_document_ids(results: Iterable[Tuple[IndexedChunk, float]]) -> List[str]:
    return list(dict.fromkeys(chunk.document_id for chunk, _ in results))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument(
        "--routing",
        choices=["all", "oracle", "lora"],
        default="all",
    )
    parser.add_argument(
        "--router-predictions",
        type=Path,
        help="Output from evaluate_enterprise_router.py; required for lora routing.",
    )
    parser.add_argument(
        "--retrieval",
        choices=["bm25", "dense", "hybrid"],
        default="hybrid",
    )
    parser.add_argument(
        "--dense-backend",
        choices=["hashing-smoke", "sentence-transformers"],
        default="hashing-smoke",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit-documents", type=int)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.routing == "lora" and not args.router_predictions:
        raise SystemExit("--router-predictions is required for --routing lora")
    grouped = load_corpus(args.corpus_dir, args.limit_documents)
    route_predictions = load_routes(args.router_predictions)
    indexes = {}
    for source, chunks in grouped.items():
        if not chunks:
            continue
        bm25 = BM25Index()
        bm25.build(chunks)
        if args.dense_backend == "sentence-transformers":
            dense = SentenceTransformerIndex(
                args.embedding_model, args.embedding_batch_size
            )
        else:
            dense = LiteDenseIndex()
        dense.build(chunks)
        indexes[source] = (bm25, dense)

    questions = list(read_jsonl(args.questions))
    if args.limit_questions:
        questions = questions[: args.limit_questions]
    rows = []
    for question in questions:
        expected = list(question.get("expected_doc_ids") or [])
        if not expected:
            continue
        if args.routing == "oracle":
            sources = list(question.get("source_types") or SOURCE_TYPES)
        elif args.routing == "lora":
            sources = route_predictions.get(str(question["question_id"]), list(SOURCE_TYPES))
        else:
            sources = list(SOURCE_TYPES)

        started = time.perf_counter()
        sparse_results = []
        dense_results = []
        for source in sources:
            if source not in indexes:
                continue
            bm25, dense = indexes[source]
            sparse_results.extend(bm25.search(question["question"], args.candidate_k))
            dense_results.extend(dense.search(question["question"], args.candidate_k))
        sparse_results.sort(key=lambda item: (-item[1], item[0].document_id))
        dense_results.sort(key=lambda item: (-item[1], item[0].document_id))
        if args.retrieval == "bm25":
            retrieved = unique_document_ids(sparse_results)[: args.top_k]
        elif args.retrieval == "dense":
            retrieved = unique_document_ids(dense_results)[: args.top_k]
        else:
            fused = reciprocal_rank_fusion(
                sparse_results[: args.candidate_k],
                dense_results[: args.candidate_k],
            )
            retrieved = list(
                dict.fromkeys(candidate.chunk.document_id for candidate in fused)
            )[: args.top_k]
        latency_ms = (time.perf_counter() - started) * 1000.0
        rows.append(
            {
                "question_id": question["question_id"],
                "question_type": question["question_type"],
                "sources": sources,
                "expected_doc_ids": expected,
                "retrieved_doc_ids": retrieved,
                "recall": recall_at_k(retrieved, expected, args.top_k),
                "reciprocal_rank": reciprocal_rank(retrieved, expected),
                "ndcg": ndcg_at_k(retrieved, expected, args.top_k),
                "latency_ms": round(latency_ms, 3),
            }
        )
    result = {
        "dataset": "EnterpriseRAG-Bench",
        "routing": args.routing,
        "retrieval": args.retrieval,
        "dense_backend": args.dense_backend,
        "embedding_model": (
            args.embedding_model
            if args.dense_backend == "sentence-transformers"
            else None
        ),
        "warning": (
            "hashing-smoke is a deterministic plumbing check, not a semantic model."
            if args.dense_backend == "hashing-smoke"
            else None
        ),
        "question_count": len(rows),
        "document_count": sum(len(chunks) for chunks in grouped.values()),
        "recall@{}".format(args.top_k): round(mean(row["recall"] for row in rows), 6),
        "mrr": round(mean(row["reciprocal_rank"] for row in rows), 6),
        "ndcg@{}".format(args.top_k): round(mean(row["ndcg"] for row in rows), 6),
        "latency_ms_mean": round(mean(row["latency_ms"] for row in rows), 3),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
