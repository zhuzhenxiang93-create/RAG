from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def padding_stats(lengths: list[int], batch_size: int, max_length: int) -> dict:
    effective = sum(lengths)
    allocated = 0
    for start in range(0, len(lengths), batch_size):
        batch = lengths[start : start + batch_size]
        padded = min(max_length, ((max(batch) + 7) // 8) * 8)
        allocated += padded * len(batch)
    return {
        "effective_tokens": effective,
        "allocated_tokens": allocated,
        "padding_tokens": allocated - effective,
        "padding_ratio": (allocated - effective) / allocated if allocated else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed_v2/train.jsonl")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output", default="artifacts/experiments/batch_benchmark_v2/padding.json")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    rows = []
    with Path(args.data).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line)["fact"])
    rng = random.Random(42)
    sample = rng.sample(rows, min(args.samples, len(rows)))
    lengths = [
        min(args.max_length, len(tokenizer.encode(text, add_special_tokens=True)))
        for text in sample
    ]
    random_order = list(lengths)
    rng.shuffle(random_order)
    length_grouped = sorted(lengths)
    static_allocated = len(lengths) * args.max_length
    result = {
        "samples": len(lengths),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "static_max_length_padding": {
            "effective_tokens": sum(lengths),
            "allocated_tokens": static_allocated,
            "padding_tokens": static_allocated - sum(lengths),
            "padding_ratio": (static_allocated - sum(lengths)) / static_allocated,
        },
        "dynamic_random_batches": padding_stats(random_order, args.batch_size, args.max_length),
        "dynamic_length_grouped_batches": padding_stats(
            length_grouped, args.batch_size, args.max_length
        ),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
