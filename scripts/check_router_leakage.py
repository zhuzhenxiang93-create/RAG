"""Standalone guard that fails if training questions overlap the benchmark."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.plugins.enterprise_router.data import (
    assert_no_benchmark_leakage,
    normalized_question_text,
    read_jsonl,
)


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalized_question_text(left).split())
    right_tokens = set(normalized_question_text(right).split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--max-token-jaccard", type=float, default=0.8)
    args = parser.parse_args()
    training = list(read_jsonl(args.train))
    benchmark = list(read_jsonl(args.benchmark))
    assert_no_benchmark_leakage(training, benchmark)
    closest = (-1.0, "", "")
    for train_row in training:
        for benchmark_row in benchmark:
            score = token_jaccard(train_row["question"], benchmark_row["question"])
            if score > closest[0]:
                closest = (
                    score,
                    str(train_row.get("question_id") or train_row.get("id") or ""),
                    str(benchmark_row.get("question_id") or benchmark_row.get("id") or ""),
                )
    result = {
        "status": "passed" if closest[0] < args.max_token_jaccard else "failed",
        "exact_id_or_normalized_text_overlap": False,
        "maximum_pairwise_token_jaccard": round(closest[0], 6),
        "threshold": args.max_token_jaccard,
        "closest_train_id": closest[1],
        "closest_benchmark_id": closest[2],
        "train_count": len(training),
        "benchmark_count": len(benchmark),
    }
    print(json.dumps(result, indent=2))
    if result["status"] == "failed":
        raise SystemExit("Potential fuzzy benchmark contamination detected")


if __name__ == "__main__":
    main()
