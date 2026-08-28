from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from legalmind.config import load_yaml
from legalmind.retrieval.fusion import deduplicate_cases, reciprocal_rank_fusion
from legalmind.retrieval.index import HybridIndex
from legalmind.retrieval.reranker import CrossEncoderReranker
from legalmind.schemas import CaseChunk, SearchHit


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def _case_ranking(hits: list[SearchHit], top_k: int) -> list[str]:
    return [hit.case_id for hit in deduplicate_cases(hits, top_k)]


def _metrics(
    rankings: list[list[str]],
    queries: list[dict],
    label_to_cases: dict[str, set[str]],
    k: int,
) -> dict[str, float | int | str]:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    hit_rates: list[float] = []
    for ranking, query in zip(rankings, queries, strict=True):
        relevant: set[str] = set()
        for label in query["labels"]["accusations"]:
            relevant.update(label_to_cases.get(label, set()))
        binary = [int(case_id in relevant) for case_id in ranking[:k]]
        found = sum(binary)
        recalls.append(found / max(len(relevant), 1))
        hit_rates.append(float(found > 0))
        first = next((rank for rank, value in enumerate(binary, start=1) if value), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(binary, 1))
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    count = max(1, len(queries))
    return {
        "relevance_definition": "proxy: any shared accusation label; not human qrels",
        f"recall@{k}": sum(recalls) / count,
        f"mrr@{k}": sum(reciprocal_ranks) / count,
        f"ndcg@{k}": sum(ndcgs) / count,
        f"hit_rate@{k}": sum(hit_rates) / count,
        "queries": len(queries),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retrieval/qwen3_pilot.yaml")
    parser.add_argument("--skip-reranker", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    chunks = [
        CaseChunk.model_validate(row)
        for row in read_jsonl(Path(config["chunks"]), int(config["max_chunks"]))
    ]
    queries = read_jsonl(Path(config["queries"]), int(config["max_queries"]))
    index = HybridIndex(
        config["embedding_model"],
        embedding_provider="local",
        query_instruction=config["query_instruction"],
        max_length=int(config["embedding_max_length"]),
        embedding_dimension=int(config["embedding_dimension"]),
    )
    build_started = time.perf_counter()
    index.build(chunks, int(config["embedding_batch_size"]))
    index.save(output / "index")
    build_seconds = time.perf_counter() - build_started
    label_to_cases: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        for label in chunk.accusations:
            label_to_cases[label].add(chunk.case_id)

    candidate_k = int(config["candidate_k"])
    fusion_top_k = int(config["fusion_top_k"])
    evaluation_k = int(config["evaluation_k"])
    rrf_k = int(config["rrf_k"])
    rankings: dict[str, list[list[str]]] = {"bm25": [], "dense": [], "hybrid": []}
    fused_candidates: list[list[SearchHit]] = []
    retrieval_latencies: list[float] = []
    for query in queries:
        query_started = time.perf_counter()
        bm25 = index.search_bm25(query["fact"], candidate_k)
        dense = index.search_vector(query["fact"], candidate_k)
        fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense}, rrf_k=rrf_k)[:fusion_top_k]
        retrieval_latencies.append(time.perf_counter() - query_started)
        rankings["bm25"].append(_case_ranking(bm25, evaluation_k))
        rankings["dense"].append(_case_ranking(dense, evaluation_k))
        rankings["hybrid"].append(_case_ranking(fused, evaluation_k))
        fused_candidates.append(fused)

    metrics = {
        name: _metrics(values, queries, label_to_cases, evaluation_k)
        for name, values in rankings.items()
    }
    reranker_seconds = None
    if not args.skip_reranker:
        reranker = CrossEncoderReranker(
            config["reranker_model"],
            max_length=int(config["reranker_max_length"]),
            batch_size=int(config["reranker_batch_size"]),
        )
        rerank_started = time.perf_counter()
        reranked: list[list[str]] = []
        for query, hits in zip(queries, fused_candidates, strict=True):
            reranked_hits = reranker.rerank(query["fact"], hits, evaluation_k)
            reranked.append(_case_ranking(reranked_hits, evaluation_k))
        reranker_seconds = time.perf_counter() - rerank_started
        metrics["hybrid_plus_pretrained_reranker"] = _metrics(
            reranked, queries, label_to_cases, evaluation_k
        )

    result = {
        "status": "completed",
        "scope": "pilot",
        "qrels_status": "automatic_charge_overlap_proxy_unreviewed",
        "embedding_model": config["embedding_model"],
        "embedding_license": config["embedding_license"],
        "reranker_model": None if args.skip_reranker else config["reranker_model"],
        "reranker_license": None if args.skip_reranker else config["reranker_license"],
        "chunks": len(chunks),
        "queries": len(queries),
        "build_seconds": build_seconds,
        "retrieval_mean_latency_seconds": sum(retrieval_latencies)
        / max(1, len(retrieval_latencies)),
        "reranker_seconds": reranker_seconds,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Pilot corpus is a deterministic prefix of the train-only case corpus.",
            "Relevance is an automatic shared-accusation proxy, not human legal judgment.",
            "The reranker is evaluated pretrained; domain hard-negative fine-tuning is not included in this run.",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "run_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
