from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from legalmind.retrieval.hard_negatives import mine_hard_negatives
from legalmind.retrieval.lexical import LexicalBM25Index
from legalmind.schemas import CaseChunk


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed_v2/train.jsonl")
    parser.add_argument("--chunks", default="data/knowledge/cases/case_chunks.jsonl")
    parser.add_argument("--output-dir", default="data/reranker")
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--negatives-per-query", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train_triplets.jsonl"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing data: {output_path}")

    with Path(args.train).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    queries = rows[: min(args.queries, len(rows))]

    chunks = []
    with Path(args.chunks).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunk = CaseChunk.model_validate_json(line)
                if chunk.source_split == "train":
                    chunks.append(chunk)
    index = LexicalBM25Index()
    index.build(chunks)
    triplets, mining_stats = mine_hard_negatives(
        queries,
        index,
        candidates_per_query=100,
        negatives_per_query=args.negatives_per_query,
    )
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in triplets:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "dataset_version": "reranker-hard-negatives-v1",
        "source_split": "train_only",
        "review_status": "unreviewed",
        "seed": args.seed,
        "requested_queries": args.queries,
        "actual_queries": len(queries),
        "triplets": len(triplets),
        "unique_queries": len({row["query_id"] for row in triplets}),
        "negative_type_distribution": dict(Counter(row["negative_type"] for row in triplets)),
        "mining_statistics": mining_stats,
        "test_rows_used": 0,
        "input_sha256": {
            "train": sha256_file(Path(args.train)),
            "chunks": sha256_file(Path(args.chunks)),
        },
        "output_sha256": sha256_file(output_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
