from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(dataset_dir: Path, output: Path, fine_outlier_threshold: int) -> dict:
    report = {
        "fine_outlier_threshold": fine_outlier_threshold,
        "splits": {},
        "note": "Outliers are retained for traceability and trained with log targets and Huber loss.",
    }
    split_hashes: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        path = dataset_dir / f"{split}.jsonl"
        hashes = set()
        outliers = []
        invalid_fines = []
        leaked_rows = []
        rows = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                fact_hash = row["cleaning_metadata"]["fact_sha256"]
                hashes.add(fact_hash)
                fine = row["labels"].get("fine")
                if not isinstance(fine, int) or fine < 0:
                    invalid_fines.append({"case_id": row["case_id"], "fine": fine})
                elif fine >= fine_outlier_threshold:
                    outliers.append(
                        {"case_id": row["case_id"], "fact_sha256": fact_hash, "fine": fine}
                    )
                if any(
                    marker in row["fact"]
                    for marker in ("判处有期徒刑", "判处无期徒刑", "判处死刑", "并处罚金")
                ):
                    leaked_rows.append({"case_id": row["case_id"], "fact_sha256": fact_hash})
        split_hashes[split] = hashes
        report["splits"][split] = {
            "rows": rows,
            "invalid_fines": invalid_fines,
            "fine_outliers": outliers,
            "residual_explicit_sentence_markers": leaked_rows,
        }
    report["cross_split_normalized_overlap"] = {
        "train_validation": len(split_hashes["train"] & split_hashes["validation"]),
        "train_test": len(split_hashes["train"] & split_hashes["test"]),
        "validation_test": len(split_hashes["validation"] & split_hashes["test"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed_v2_1_1"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/data_v2_1_1/sentencing_data_audit.json")
    )
    parser.add_argument("--fine-outlier-threshold", type=int, default=100_000_000)
    args = parser.parse_args()
    report = audit(args.dataset_dir, args.output, args.fine_outlier_threshold)
    summary = {
        "fine_outlier_threshold": report["fine_outlier_threshold"],
        "cross_split_normalized_overlap": report["cross_split_normalized_overlap"],
        "splits": {
            split: {
                "rows": values["rows"],
                "invalid_fines": len(values["invalid_fines"]),
                "fine_outliers": len(values["fine_outliers"]),
                "residual_explicit_sentence_markers": len(
                    values["residual_explicit_sentence_markers"]
                ),
            }
            for split, values in report["splits"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
