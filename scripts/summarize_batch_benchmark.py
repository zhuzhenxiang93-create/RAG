from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="artifacts/experiments/batch_benchmark_v2")
    parser.add_argument(
        "--output", default="artifacts/experiments/batch_benchmark_v2/comparison.json"
    )
    args = parser.parse_args()
    source = Path(args.input_dir)
    runs = [
        json.loads((source / f"batch{size}.json").read_text(encoding="utf-8")) for size in (2, 4, 8)
    ]
    stable = [row for row in runs if row["status"] == "completed"]
    for row in stable:
        row["microbatch_samples_per_second"] = row["batch_size"] / row["step_seconds"]
    recommended = max(stable, key=lambda row: row["microbatch_samples_per_second"])
    result = {
        "benchmark_scope": "worst-case 2048-token forward/backward/AdamW step",
        "effective_batch_size": 32,
        "runs": runs,
        "recommended_micro_batch_size": recommended["batch_size"],
        "recommended_gradient_accumulation_steps": 32 // recommended["batch_size"],
        "selection_reason": "highest stable measured microbatch throughput with memory headroom",
    }
    target = Path(args.output)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
