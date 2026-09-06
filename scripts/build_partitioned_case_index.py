from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalmind.retrieval.partitioned import ChargePartitionedRetriever
from legalmind.schemas import CaseChunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("artifacts/experiments/bm25_sparse64_eval_v2/index/chunks.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/indexes/case_retrieval_by_charge"),
    )
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--candidate-multiplier", type=int, default=20)
    args = parser.parse_args()
    with args.chunks.open(encoding="utf-8") as handle:
        chunks = [CaseChunk.model_validate_json(line) for line in handle if line.strip()]
    retriever = ChargePartitionedRetriever(
        min_score=args.min_score, candidate_multiplier=args.candidate_multiplier
    )
    counts = retriever.build(chunks)
    retriever.save(args.output_dir)
    print(
        json.dumps(
            {
                "input_chunks": len(chunks),
                "partitions": len(counts),
                "indexed_chunks": sum(counts.values()),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
