"""Prepare contamination-safe LoRA train/validation data and benchmark labels."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.plugins.enterprise_router.data import (
    assert_no_benchmark_leakage,
    normalized_training_row,
    read_jsonl,
    write_jsonl,
)


def digest_files(paths):
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a separately generated EnterpriseRAG question set into router "
            "training data. Official Redwood questions are benchmark-only."
        )
    )
    parser.add_argument(
        "--train-questions",
        type=Path,
        required=True,
        help="questions.jsonl from a separately generated training company",
    )
    parser.add_argument(
        "--benchmark-questions",
        type=Path,
        required=True,
        help="official Redwood questions.jsonl; used only for leakage checks/evaluation",
    )
    parser.add_argument("--output", type=Path, default=Path("data/enterprise_router"))
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_questions.resolve() == args.benchmark_questions.resolve():
        raise SystemExit(
            "Refusing to train on the benchmark file. Generate a separate company/question set."
        )
    if not 0.0 < args.validation_ratio < 0.5:
        raise SystemExit("--validation-ratio must be between 0 and 0.5")

    training_rows = list(read_jsonl(args.train_questions))
    benchmark_rows = list(read_jsonl(args.benchmark_questions))
    if len(training_rows) < 2:
        raise SystemExit("At least two independent training questions are required")
    assert_no_benchmark_leakage(training_rows, benchmark_rows)

    randomizer = random.Random(args.seed)
    randomizer.shuffle(training_rows)
    validation_count = max(1, round(len(training_rows) * args.validation_ratio))
    validation = training_rows[:validation_count]
    train = training_rows[validation_count:]
    benchmark = [normalized_training_row(row) for row in benchmark_rows]

    output = args.output
    counts = {
        "train": write_jsonl(output / "train.jsonl", map(normalized_training_row, train)),
        "validation": write_jsonl(
            output / "validation.jsonl", map(normalized_training_row, validation)
        ),
        "benchmark": write_jsonl(output / "benchmark.jsonl", benchmark),
    }
    manifest = {
        "dataset": "EnterpriseRAG-Bench-compatible router data",
        "official_benchmark_role": "evaluation_only",
        "train_source": str(args.train_questions),
        "benchmark_source": str(args.benchmark_questions),
        "source_sha256": digest_files(
            [args.train_questions, args.benchmark_questions]
        ),
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "counts": counts,
        "leakage_check": "passed",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

