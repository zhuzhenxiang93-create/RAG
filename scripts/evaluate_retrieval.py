from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    values: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            values[str(row["query_id"])][str(row["case_id"])] = int(row["relevance"])
    return values


def load_run(path: Path) -> dict[str, list[str]]:
    values: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            values[str(row["query_id"])].append((int(row["rank"]), str(row["case_id"])))
    return {query_id: [case_id for _, case_id in sorted(rows)] for query_id, rows in values.items()}


def dcg(relevances: list[int], k: int) -> float:
    return sum((2**rel - 1) / math.log2(index + 2) for index, rel in enumerate(relevances[:k]))


def evaluate(
    qrels: dict[str, dict[str, int]], run: dict[str, list[str]], k: int
) -> dict[str, float]:
    recalls, reciprocal_ranks, ndcgs = [], [], []
    for query_id, judgments in qrels.items():
        relevant = {case_id for case_id, rel in judgments.items() if rel > 0}
        ranking = run.get(query_id, [])[:k]
        recalls.append(len(relevant.intersection(ranking)) / max(len(relevant), 1))
        first = next((rank for rank, case_id in enumerate(ranking, 1) if case_id in relevant), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        observed = [judgments.get(case_id, 0) for case_id in ranking]
        ideal = sorted(judgments.values(), reverse=True)
        denominator = dcg(ideal, k)
        ndcgs.append(dcg(observed, k) / denominator if denominator else 0.0)
    count = max(len(qrels), 1)
    return {
        f"recall@{k}": sum(recalls) / count,
        f"mrr@{k}": sum(reciprocal_ranks) / count,
        f"ndcg@{k}": sum(ndcgs) / count,
        "queries": float(len(qrels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="artifacts/results/retrieval_metrics.json")
    args = parser.parse_args()
    metrics = evaluate(load_qrels(Path(args.qrels)), load_run(Path(args.run)), args.k)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
