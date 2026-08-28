from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from legalmind.generation.sft_data import build_sft_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed_v2")
    parser.add_argument("--output-dir", default="data/sft")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-validation", type=int)
    args = parser.parse_args()
    source = Path(args.data_dir)
    output = Path(args.output_dir)
    train = output / "train.jsonl"
    validation = output / "validation.jsonl"
    counts = {
        "train": build_sft_file(source / "train.jsonl", train, args.max_train),
        "validation": build_sft_file(source / "validation.jsonl", validation, args.max_validation),
    }
    manifest = {
        "dataset": "LegalMind structured generation SFT",
        "construction": "deterministic_from_training_labels",
        "review_status": "unreviewed",
        "source_splits": {"train": "train", "validation": "validation"},
        "counts": counts,
        "sha256": {"train": sha256_file(train), "validation": sha256_file(validation)},
        "known_limitation": "legacy source lacks verified statute labels; article targets remain empty",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "rejected.jsonl").touch(exist_ok=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
