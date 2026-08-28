from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from legalmind.retrieval.hard_negatives import mine_hard_negatives
from legalmind.retrieval.lexical import LexicalBM25Index
from legalmind.schemas import CaseChunk


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def evaluate_charge_overlap_proxy(
    queries: list[dict], index: LexicalBM25Index, top_k: int = 10
) -> dict:
    label_to_cases: dict[str, set[str]] = defaultdict(set)
    for chunk in index.chunks:
        for label in chunk.accusations:
            label_to_cases[label].add(chunk.case_id)
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    hit_rates: list[float] = []
    latencies: list[float] = []
    for query in queries:
        relevant = set().union(*(label_to_cases[label] for label in query["labels"]["accusations"]))
        started = time.perf_counter()
        hits = index.search(query["fact"], top_k * 3)
        latencies.append(time.perf_counter() - started)
        ranking = []
        seen = set()
        for hit in hits:
            if hit.case_id not in seen:
                ranking.append(hit.case_id)
                seen.add(hit.case_id)
            if len(ranking) >= top_k:
                break
        binary = [int(case_id in relevant) for case_id in ranking]
        found = sum(binary)
        recalls.append(found / max(len(relevant), 1))
        hit_rates.append(float(found > 0))
        first = next((rank for rank, value in enumerate(binary, start=1) if value), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(binary, start=1))
        ideal_count = min(top_k, len(relevant))
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    count = max(len(queries), 1)
    return {
        "relevance_definition": "proxy: any shared accusation label; not human qrels",
        f"recall@{top_k}": sum(recalls) / count,
        f"mrr@{top_k}": sum(reciprocal_ranks) / count,
        f"ndcg@{top_k}": sum(ndcgs) / count,
        f"hit_rate@{top_k}": sum(hit_rates) / count,
        "queries": len(queries),
        "latency_seconds": {
            "mean": sum(latencies) / count,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/knowledge/cases/case_chunks.jsonl")
    parser.add_argument("--queries", default="data/processed_v2/validation.jsonl")
    parser.add_argument("--output-dir", default="artifacts/experiments/bm25_v2")
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--max-queries", type=int, default=200)
    args = parser.parse_args()
    started = time.perf_counter()
    chunk_rows = read_jsonl(Path(args.chunks), args.max_chunks)
    chunks = [CaseChunk.model_validate(row) for row in chunk_rows]
    index = LexicalBM25Index()
    index.build(chunks)
    output = Path(args.output_dir)
    index.save(output / "index")
    queries = read_jsonl(Path(args.queries), args.max_queries)
    negatives, stats = mine_hard_negatives(queries, index)
    metrics = evaluate_charge_overlap_proxy(queries, index)
    with (output / "hard_negatives.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in negatives:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "hard_negative_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "retrieval_baseline.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "status": "completed",
        "index": "sparse_inverted_BM25",
        "chunks": len(chunks),
        "queries": len(queries),
        "hard_negatives": stats,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
