from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

from legalmind.data.normalize import normalize_record, normalized_text_hash


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_jsonl(path: Path) -> dict:
    rows = errors = empty = multi = 0
    labels: Counter[str] = Counter()
    schemas: Counter[tuple[str, ...]] = Counter()
    normalized_hashes: set[str] = set()
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            try:
                raw = json.loads(line)
                schemas[tuple(sorted(raw))] += 1
                case = normalize_record(raw, source_split="audit")
                labels.update(case.accusations)
                empty += int(not case.accusations)
                multi += int(len(case.accusations) > 1)
                normalized_hashes.add(normalized_text_hash(case.fact))
            except (ValueError, TypeError, json.JSONDecodeError):
                errors += 1
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
        "json_errors": errors,
        "schema_variants": len(schemas),
        "unique_labels": len(labels),
        "empty_label_rows": empty,
        "multi_label_rows": multi,
        "normalized_duplicate_rows": rows - len(normalized_hashes),
        "top_labels": labels.most_common(20),
        "tail_labels": sorted(labels.items(), key=lambda item: item[1])[:20],
        "_hashes": normalized_hashes,
    }


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def write_markdown(path: Path, title: str, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {title}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/audit")
    args = parser.parse_args()
    output = Path(args.output_dir)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "omp_num_threads_before_run": os.getenv("OMP_NUM_THREADS"),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "disk": command_output(["df", "-h", "/root/autodl-tmp"]),
    }
    repository = {
        "root": str(Path.cwd()),
        "has_git": Path(".git").exists(),
        "source_commit": Path(".source-commit").read_text(encoding="utf-8").strip()
        if Path(".source-commit").exists()
        else None,
        "source_python_files": len(list(Path("src/legalmind").rglob("*.py"))),
        "test_files": len(list(Path("tests").glob("test_*.py"))),
    }
    paths = [
        Path("data/train_data.jsonl"),
        Path("data/local_legacy/rest_data.jsonl"),
        Path("data/local_legacy/test_data.jsonl"),
        Path("artifacts/datasets/cail2018_multilabel/train.jsonl"),
        Path("artifacts/datasets/cail2018_multilabel/validation.jsonl"),
        Path("artifacts/datasets/cail2018_multilabel/test.jsonl"),
    ]
    audits = {str(path): audit_jsonl(path) for path in paths if path.exists()}
    overlaps = {}
    names = list(audits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = len(audits[left]["_hashes"] & audits[right]["_hashes"])
            if overlap:
                overlaps[f"{left}::{right}"] = overlap
    for value in audits.values():
        value.pop("_hashes")
    data = {
        "files": audits,
        "normalized_cross_file_overlaps": overlaps,
        "provenance_warning": "legacy local files have no verified original download URL or license",
        "rest_data_decision": "audit_only_pending_provenance_and_distribution_review",
    }
    historical = {
        "classification_10k_metrics": json.loads(
            Path("artifacts/results/classifier_10k_2k/metrics.json").read_text(encoding="utf-8")
        ),
        "classification_10k_manifest": json.loads(
            Path("artifacts/models/qwen3_4b_qlora_10k/run_manifest.json").read_text(
                encoding="utf-8"
            )
        ),
        "interpretation": "verified historical 10K coverage experiment; not a full-data result",
    }
    write_markdown(output / "environment_audit.md", "Environment audit", environment)
    write_markdown(output / "repository_audit.md", "Repository audit", repository)
    write_markdown(output / "data_audit.md", "Data audit", data)
    write_markdown(output / "historical_experiments.md", "Historical experiments", historical)
    print(
        json.dumps(
            {"environment": environment, "repository": repository, "data": data},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
