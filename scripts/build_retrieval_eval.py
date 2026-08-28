from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from legalmind.retrieval.evaluation_data import (
    build_silver_candidates,
    select_diverse_queries,
    sha256_file,
    stable_json_line,
)
from legalmind.retrieval.lexical import LexicalBM25Index
from legalmind.schemas import CaseChunk


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(stable_json_line(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed_v2/train.jsonl")
    parser.add_argument("--validation", default="data/processed_v2/validation.jsonl")
    parser.add_argument("--chunks", default="data/knowledge/cases/case_chunks.jsonl")
    parser.add_argument("--output-dir", default="data/retrieval_eval")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite retrieval evaluation data: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(Path(args.train))
    validation_rows = read_jsonl(Path(args.validation))
    support: Counter[str] = Counter()
    for row in train_rows:
        support.update(row["labels"]["accusations"])
    queries = select_diverse_queries(validation_rows, dict(support), args.queries, args.seed)

    chunks = []
    with Path(args.chunks).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(CaseChunk.model_validate_json(line))
    index = LexicalBM25Index()
    index.build(chunks)
    candidates, qrels, review_rows, stats = build_silver_candidates(queries, index)

    paths = {
        "queries": output / "queries.jsonl",
        "candidate_pool": output / "candidate_pool.jsonl",
        "silver_qrels": output / "silver_qrels.jsonl",
        "reviewed_qrels": output / "reviewed_qrels.jsonl",
    }
    write_jsonl(paths["queries"], queries)
    write_jsonl(paths["candidate_pool"], candidates)
    write_jsonl(paths["silver_qrels"], qrels)
    paths["reviewed_qrels"].write_text("", encoding="utf-8")

    review_path = output / "manual_review_queue.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(review_rows[0]) if review_rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(review_rows)

    manifest = {
        "dataset_version": "retrieval-eval-v1",
        "seed": args.seed,
        "query_source_split": "validation_only",
        "candidate_source_split": "train_only",
        "reviewed_qrels": 0,
        "manual_review_status": "pending",
        "license_status": "inherits_CAIL2018_legacy_local_unverified",
        **stats,
    }
    files = [*paths.values(), review_path]
    manifest["file_sha256"] = {path.name: sha256_file(path) for path in files}
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
