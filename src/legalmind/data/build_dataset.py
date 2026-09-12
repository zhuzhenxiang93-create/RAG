from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from legalmind.config import load_yaml
from legalmind.data.contracts import ProcessedCase
from legalmind.data.deduplicate import exact_fingerprint, near_duplicate_candidates
from legalmind.data.leakage import strip_target_leakage
from legalmind.data.loader import iter_cases
from legalmind.data.normalize import normalize_text, normalized_text_hash
from legalmind.data.prepare import canonicalize_labels, is_low_information_fact
from legalmind.data.split import multilabel_group_split
from legalmind.data.statistics import frequency_bands, split_statistics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def penalty_labels(value: dict | None) -> dict:
    value = value or {}
    months = value.get("imprisonment_months", value.get("imprisonment"))
    fine = value.get("fine")
    return {
        "imprisonment_months": int(months) if str(months).lstrip("-").isdigit() else None,
        "fine": int(fine) if str(fine).strip().isdigit() else None,
        "life_imprisonment": bool(value.get("life_imprisonment", False)),
        "death_penalty": bool(value.get("death_penalty", False)),
    }


def penalty_signature(value: dict | None) -> tuple[int | None, bool, bool, int | None]:
    labels = penalty_labels(value)
    return (
        labels["imprisonment_months"],
        labels["life_imprisonment"],
        labels["death_penalty"],
        labels["fine"],
    )


def representative_penalty(values: list[dict | None]) -> tuple[dict, bool]:
    """Select a deterministic modal outcome while making disagreements auditable."""
    signatures = [penalty_signature(value) for value in values]
    counts = Counter(signatures)
    selected = sorted(counts, key=lambda item: (-counts[item], str(item)))[0]
    return {
        "imprisonment_months": selected[0],
        "life_imprisonment": selected[1],
        "death_penalty": selected[2],
        "fine": selected[3],
    }, len(counts) > 1


def build(config_path: str | Path) -> dict:
    config = load_yaml(config_path)
    raw_path = Path(config["raw_path"])
    output_dir = Path(config["output_dir"])
    reports_dir = Path(config["reports_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    counters: Counter[str] = Counter()
    merged: dict[str, dict] = {}
    exact_seen: set[str] = set()
    for source_index, case in enumerate(iter_cases(raw_path, "legacy_train"), start=1):
        counters["raw_rows"] += 1
        fact = normalize_text(case.fact)
        accusations = canonicalize_labels(case.accusations)
        if not fact or not accusations:
            counters["empty_or_unlabelled_rows"] += 1
            continue
        if len(fact) < int(config.get("min_text_chars", 20)):
            counters["short_rows"] += 1
            continue
        if is_low_information_fact(fact, int(config.get("max_boilerplate_chars", 120))):
            counters["low_information_rows"] += 1
            continue
        exact = exact_fingerprint(case.fact)
        if exact in exact_seen:
            counters["exact_duplicate_rows"] += 1
        exact_seen.add(exact)
        leakage = strip_target_leakage(fact)
        clean_hash = normalized_text_hash(leakage.text)
        if clean_hash in merged:
            counters["normalized_duplicate_rows"] += 1
            previous = merged[clean_hash]
            previous["accusations"].update(accusations)
            previous["articles"].update(case.relevant_articles)
            previous["penalties"].append(case.penalty)
            continue
        merged[clean_hash] = {
            "source_index": source_index,
            "fact": leakage.text,
            "accusations": set(accusations),
            "articles": set(case.relevant_articles),
            "penalties": [case.penalty],
            "leakage": leakage,
            "dedup_group_id": f"DG-{clean_hash[:20]}",
            "fact_sha256": clean_hash,
        }

    label_counts = Counter(label for row in merged.values() for label in row["accusations"])
    labels = sorted(label_counts)
    mapping = {label: index for index, label in enumerate(labels)}
    records: list[dict] = []
    penalty_conflicts: list[dict] = []
    for row in merged.values():
        accusations = sorted(row["accusations"])
        penalties, has_penalty_conflict = representative_penalty(row["penalties"])
        if has_penalty_conflict:
            counters["duplicate_penalty_conflict_groups"] += 1
            penalty_conflicts.append(
                {
                    "dedup_group_id": row["dedup_group_id"],
                    "fact_sha256": row["fact_sha256"],
                    "observations": len(row["penalties"]),
                    "outcomes": [
                        {
                            "imprisonment_months": signature[0],
                            "life_imprisonment": signature[1],
                            "death_penalty": signature[2],
                            "fine": signature[3],
                            "count": count,
                        }
                        for signature, count in sorted(
                            Counter(penalty_signature(value) for value in row["penalties"]).items(),
                            key=lambda item: str(item[0]),
                        )
                    ],
                    "selected": penalties,
                }
            )
        record = ProcessedCase(
            case_id=f"CAIL-{row['fact_sha256'][:16]}",
            source_dataset=config.get("source_dataset", "CAIL2018_legacy_local"),
            source_split="pending",
            fact=row["fact"],
            labels={
                "accusations": accusations,
                "relevant_articles": sorted(row["articles"]),
                **penalties,
            },
            accusation_ids=[mapping[label] for label in accusations],
            cleaning_metadata={
                "normalized": True,
                "target_leakage_removed": row["leakage"].removed,
                "leakage_types": list(row["leakage"].leakage_types),
                "dedup_group_id": row["dedup_group_id"],
                "fact_sha256": row["fact_sha256"],
            },
            length_metadata={"characters": len(row["fact"]), "tokens": None},
        ).model_dump()
        records.append(record)

    splits = multilabel_group_split(
        records,
        len(mapping),
        float(config.get("validation_ratio", 0.1)),
        float(config.get("test_ratio", 0.1)),
        int(config.get("seed", 42)),
    )
    for split, rows in splits.items():
        write_jsonl(rows, output_dir / f"{split}.jsonl")
    (output_dir / "label_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    split_hashes = {
        path.name: sha256_file(path)
        for path in [
            output_dir / "train.jsonl",
            output_dir / "validation.jsonl",
            output_dir / "test.jsonl",
            output_dir / "label_mapping.json",
        ]
    }
    statistics = {name: split_statistics(rows, len(mapping)) for name, rows in splits.items()}
    manifest = {
        "dataset_version": config.get("dataset_version", "2.0.0"),
        "source_dataset": config.get("source_dataset", "CAIL2018_legacy_local"),
        "source_status": config.get("source_status", "legacy_local_file_unverified"),
        "source_sha256": sha256_file(raw_path),
        "seed": int(config.get("seed", 42)),
        "split_strategy": "normalized_dedup_then_iterative_multilabel_group_split",
        "sizes": {name: len(rows) for name, rows in splits.items()},
        "num_labels": len(mapping),
        "filtering": dict(counters),
        "statistics": statistics,
        "frequency_bands": frequency_bands(label_counts),
        "output_sha256": split_hashes,
        "quality": {
            "exact_cross_split_overlap": 0,
            "normalized_cross_split_overlap": 0,
            "dedup_group_cross_split_overlap": 0,
            "review_status": "unreviewed",
        },
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (output_dir / "dataset_manifest.json").write_text(manifest_text, encoding="utf-8")
    (output_dir / "file_hashes.json").write_text(
        json.dumps(split_hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (reports_dir / "label_distribution.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "label_id", "total_count"])
        for label, label_id in mapping.items():
            writer.writerow([label, label_id, label_counts[label]])

    near_sample = int(config.get("near_duplicate_scan_rows", 20_000))
    near_rows = [(row["case_id"], row["fact"]) for row in records[:near_sample]]
    near_pairs = near_duplicate_candidates(
        near_rows,
        threshold=int(config.get("simhash_hamming_threshold", 3)),
        max_pairs=int(config.get("max_near_duplicate_pairs", 5_000)),
    )
    duplicate_report = {
        "raw_rows": counters["raw_rows"],
        "exact_duplicate_rows": counters["exact_duplicate_rows"],
        "normalized_duplicate_rows": counters["normalized_duplicate_rows"],
        "near_duplicate_algorithm": "SimHash64 character-trigram, four 16-bit LSH bands",
        "near_duplicate_threshold": int(config.get("simhash_hamming_threshold", 3)),
        "near_duplicate_scan_rows": min(near_sample, len(records)),
        "near_duplicate_candidate_pairs": len(near_pairs),
        "near_duplicate_pairs_truncated": len(near_pairs)
        >= int(config.get("max_near_duplicate_pairs", 5_000)),
        "cross_split_exact": 0,
        "cross_split_normalized": 0,
    }
    (reports_dir / "duplicate_report.json").write_text(
        json.dumps(duplicate_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "duplicate_penalty_conflicts.json").write_text(
        json.dumps(penalty_conflicts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "near_duplicate_candidates.json").write_text(
        json.dumps(near_pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "data_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "length_distribution.json").write_text(
        json.dumps(
            {name: value["fact_characters"] for name, value in statistics.items()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (reports_dir / "leakage_report.json").write_text(
        json.dumps(
            {
                "target_leakage_removed_rows": sum(
                    value["target_leakage_removed_rows"] for value in statistics.values()
                ),
                "cross_split_normalized_overlap": 0,
                "method": "rule-based charge-and-sentencing-conclusion masking before split",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (reports_dir / "split_report.json").write_text(
        json.dumps(
            {
                "sizes": manifest["sizes"],
                "strategy": manifest["split_strategy"],
                "leakage": manifest["quality"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/processed_v2.yaml")
    args = parser.parse_args()
    print(json.dumps(build(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
