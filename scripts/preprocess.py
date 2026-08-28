from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from legalmind.config import load_yaml
from legalmind.data.labels import build_label_mapping, save_label_mapping
from legalmind.data.loader import iter_cases
from legalmind.data.prepare import (
    canonicalize_labels,
    is_low_information_fact,
    merge_case_labels,
)
from legalmind.schemas import CaseRecord


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_multilabel(
    records: list[dict],
    num_labels: int,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
):
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    labels = np.zeros((len(records), num_labels), dtype=np.int8)
    for row, record in enumerate(records):
        labels[row, record["accusation_ids"]] = 1
    indices = np.arange(len(records))
    first = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_ratio + test_ratio,
        random_state=seed,
    )
    train_indices, held_indices = next(first.split(indices, labels))
    held_labels = labels[held_indices]
    test_share = test_ratio / (validation_ratio + test_ratio)
    second = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_share,
        random_state=seed + 1,
    )
    validation_local, test_local = next(second.split(held_indices, held_labels))
    return (
        [records[index] for index in train_indices],
        [records[held_indices[index]] for index in validation_local],
        [records[held_indices[index]] for index in test_local],
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_statistics(records: list[dict], num_labels: int) -> dict:
    counts = Counter(
        label_id for record in records for label_id in record.get("accusation_ids", [])
    )
    lengths = sorted(len(record["fact"]) for record in records)

    def percentile(fraction: float) -> int:
        if not lengths:
            return 0
        return lengths[min(len(lengths) - 1, int((len(lengths) - 1) * fraction))]

    return {
        "rows": len(records),
        "multi_label_rows": sum(len(record["accusation_ids"]) > 1 for record in records),
        "labels_present": len(counts),
        "labels_missing": sorted(set(range(num_labels)) - set(counts)),
        "label_frequency_min": min(counts.values(), default=0),
        "label_frequency_max": max(counts.values(), default=0),
        "fact_chars": {
            "min": lengths[0] if lengths else 0,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": lengths[-1] if lengths else 0,
        },
    }


def assert_disjoint_splits(splits: dict[str, list[dict]]) -> None:
    fingerprints = {
        name: {hashlib.sha1(record["fact"].encode("utf-8")).digest() for record in records}
        for name, records in splits.items()
    }
    names = list(fingerprints)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = fingerprints[left] & fingerprints[right]
            if overlap:
                raise ValueError(f"Data leakage: {left} and {right} share {len(overlap)} facts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    raw_train = Path(config["raw"]["train"])
    cases = list(iter_cases(raw_train, "legacy_train"))
    max_samples = config.get("max_train_samples")
    if max_samples:
        random.shuffle(cases)
        cases = cases[: int(max_samples)]

    counters = Counter(raw_rows=len(cases))
    merged_cases: dict[str, CaseRecord] = {}
    min_chars = int(config.get("min_text_chars", 20))
    max_boilerplate_chars = int(config.get("max_boilerplate_chars", 120))
    for case in cases:
        case.accusations = canonicalize_labels(case.accusations)
        if not case.accusations:
            counters["empty_label_rows"] += 1
            continue
        if len(case.fact) < min_chars:
            counters["short_text_rows"] += 1
            continue
        if is_low_information_fact(case.fact, max_chars=max_boilerplate_chars):
            counters["low_information_rows"] += 1
            continue
        existing = merged_cases.get(case.fact)
        if existing is not None:
            counters["duplicate_rows"] += 1
            counters["conflicting_duplicate_rows"] += int(merge_case_labels(existing, case))
            continue
        merged_cases[case.fact] = case

    cleaned_cases = list(merged_cases.values())
    mapping = build_label_mapping([case.accusations for case in cleaned_cases])
    records = []
    for case in cleaned_cases:
        case.accusation_ids = [mapping[label] for label in case.accusations]
        records.append(case.model_dump())
    train, validation, test = split_multilabel(
        records,
        len(mapping),
        float(config.get("validation_ratio", 0.1)),
        float(config.get("test_ratio", 0.1)),
        seed,
    )

    output_dir = Path(config["output_dir"])
    staging_dir = output_dir.with_name(f"{output_dir.name}.staging")
    if staging_dir.exists():
        raise FileExistsError(f"Remove or inspect stale staging directory first: {staging_dir}")
    splits = {"train": train, "validation": validation, "test": test}
    assert_disjoint_splits(splits)
    for split, values in splits.items():
        for value in values:
            value["source_split"] = split
        write_jsonl(values, staging_dir / f"{split}.jsonl")

    save_label_mapping(mapping, staging_dir / "label_mapping.json")
    output_hashes = {
        f"{split}.jsonl": sha256_file(staging_dir / f"{split}.jsonl") for split in splits
    }
    output_hashes["label_mapping.json"] = sha256_file(staging_dir / "label_mapping.json")
    statistics = {split: split_statistics(values, len(mapping)) for split, values in splits.items()}
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "seed": seed,
        "input": {
            "path": str(raw_train),
            "bytes": raw_train.stat().st_size,
            "sha256": sha256_file(raw_train),
            "provenance_status": config.get("provenance_status", "unverified"),
            "source_url": config.get("source_url"),
        },
        "filtering": {
            "min_text_chars": min_chars,
            "max_boilerplate_chars": max_boilerplate_chars,
            **dict(counters),
            "clean_unique_cases": len(cleaned_cases),
        },
        "sizes": {split: len(values) for split, values in splits.items()},
        "num_labels": len(mapping),
        "split_strategy": "iterative_multilabel_stratification",
        "validation_ratio": float(config.get("validation_ratio", 0.1)),
        "test_ratio": float(config.get("test_ratio", 0.1)),
        "exact_text_overlap": 0,
        "statistics": statistics,
        "output_sha256": output_hashes,
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    (staging_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    if output_dir.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_dir = output_dir.with_name(f"{output_dir.name}.previous-{timestamp}")
        output_dir.replace(backup_dir)
    staging_dir.replace(output_dir)
    save_label_mapping(mapping, config["label_mapping"])
    manifest_path = config.get("manifest_path")
    if manifest_path:
        manifest_target = Path(manifest_path)
        manifest_target.parent.mkdir(parents=True, exist_ok=True)
        manifest_target.write_text(manifest_text, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
