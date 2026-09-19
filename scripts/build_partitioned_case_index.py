from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalmind.data.label_overrides import apply_label_overrides, load_label_overrides
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
    parser.add_argument(
        "--label-overrides",
        type=Path,
        help="JSONL review decisions; only status=approved records are applied.",
    )
    args = parser.parse_args()
    with args.chunks.open(encoding="utf-8") as handle:
        chunks = [CaseChunk.model_validate_json(line) for line in handle if line.strip()]
    override_audit = {"configured": False, "approved": 0, "applied_chunks": 0}
    if args.label_overrides:
        overrides = load_label_overrides(args.label_overrides)
        chunks, override_audit = apply_label_overrides(chunks, overrides)
    retriever = ChargePartitionedRetriever(
        min_score=args.min_score, candidate_multiplier=args.candidate_multiplier
    )
    counts = retriever.build(chunks)
    retriever.save(args.output_dir)
    (args.output_dir / "label_overrides_audit.json").write_text(
        json.dumps(override_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "input_chunks": len(chunks),
                "partitions": len(counts),
                "indexed_chunks": sum(counts.values()),
                "output_dir": str(args.output_dir),
                "label_overrides": override_audit,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
