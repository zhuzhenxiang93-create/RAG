from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def nested(value: dict, *keys: str):
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/experiments")
    parser.add_argument("--output", default="artifacts/experiments/experiment_index.csv")
    args = parser.parse_args()
    rows = []
    for manifest_path in sorted(Path(args.root).glob("*/run_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        metrics = manifest.get("metrics", {})
        status = manifest.get("status")
        if not status and manifest.get("final_validation_metrics"):
            status = "completed"
        micro_f1 = metrics.get(
            "micro_f1",
            metrics.get(
                "accusation_micro_f1",
                manifest.get("best_micro_f1", manifest.get("best_metric", "")),
            ),
        )
        rows.append(
            {
                "experiment": manifest_path.parent.name,
                "status": status or "unknown",
                "scope": manifest.get("scope", manifest.get("experiment_type", "unknown")),
                "model": manifest.get("model", manifest.get("embedding_model", "")),
                "train_rows": manifest.get("train_rows", ""),
                "validation_rows": manifest.get("validation_rows", manifest.get("queries", "")),
                "micro_f1": micro_f1,
                "elapsed_seconds": manifest.get("elapsed_seconds", ""),
                "peak_gpu_memory_gib": manifest.get("peak_gpu_memory_gib", ""),
                "data_train_sha256": nested(manifest, "data_sha256", "train")
                or nested(manifest, "data_hashes", "train")
                or "",
                "manifest": str(manifest_path),
            }
        )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["experiment", "status", "manifest"]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"experiments": len(rows), "output": str(target)}, indent=2))


if __name__ == "__main__":
    main()
