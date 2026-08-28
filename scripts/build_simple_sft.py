from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from legalmind.generation.simple_sft import build_simple_sft_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed_v2")
    parser.add_argument("--output-dir", default="data/sft_v3_1")
    parser.add_argument("--train-rows", type=int, default=4000)
    parser.add_argument("--validation-rows", type=int, default=500)
    parser.add_argument("--train-uncertainty-rows", type=int, default=500)
    parser.add_argument("--validation-uncertainty-rows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = Path(args.data_dir)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite SFT data: {output}")
    output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((source / "label_mapping.json").read_text(encoding="utf-8"))
    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    train_report = build_simple_sft_file(
        source / "train.jsonl",
        train_path,
        args.train_rows,
        len(mapping),
        args.seed,
        args.train_uncertainty_rows,
    )
    validation_report = build_simple_sft_file(
        source / "validation.jsonl",
        validation_path,
        args.validation_rows,
        len(mapping),
        args.seed,
        args.validation_uncertainty_rows,
    )
    manifest = {
        "dataset_version": "sft-v3.1-simplified-json-with-abstention",
        "schema": [
            "candidate_accusations",
            "key_facts",
            "confidence",
            "requires_manual_review",
        ],
        "seed": args.seed,
        "train": train_report,
        "validation": validation_report,
        "test_rows_used": 0,
        "review_status": "unreviewed",
        "file_sha256": {
            "train.jsonl": sha256_file(train_path),
            "validation.jsonl": sha256_file(validation_path),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "rejected.jsonl").write_text("", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
