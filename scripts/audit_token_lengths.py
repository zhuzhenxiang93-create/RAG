from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalmind.data.loader import iter_jsonl
from legalmind.data.token_stats import summarize_token_lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="artifacts/datasets/cail2018_multilabel")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", default="artifacts/results/token_length_audit.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    report = {"tokenizer": args.tokenizer, "splits": {}}
    dataset_dir = Path(args.dataset_dir)
    for split in ("train", "validation", "test"):
        lengths: list[int] = []
        batch: list[str] = []
        for record in iter_jsonl(dataset_dir / f"{split}.jsonl"):
            batch.append(record["fact"])
            if len(batch) >= args.batch_size:
                encoded = tokenizer(batch, add_special_tokens=True, return_length=True)
                lengths.extend(int(value) for value in encoded["length"])
                batch.clear()
        if batch:
            encoded = tokenizer(batch, add_special_tokens=True, return_length=True)
            lengths.extend(int(value) for value in encoded["length"])
        report["splits"][split] = summarize_token_lengths(lengths)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
