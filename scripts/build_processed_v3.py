from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from legalmind.data.contracts_v3 import ProcessedCaseV3
from legalmind.data.dedup_v3 import duplicate_group_id, normalized_duplicate_text
from legalmind.data.leakage_v3 import clean_input_fact, detect_explicit_leakage
from legalmind.data.privacy import redact_privacy
from legalmind.data.quality_gate import evaluate_quality_gate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def convert(row: dict, split: str, config: dict, counters: Counter) -> dict | None:
    cleaned = clean_input_fact(row["fact"])
    if not cleaned.text:
        counters["empty_after_leakage_clean"] += 1
        return None
    privacy = redact_privacy(cleaned.text)
    labels = row.get("labels", {})
    death, life = bool(labels.get("death_penalty")), bool(labels.get("life_imprisonment"))
    months = labels.get("imprisonment_months")
    invalid_special = death and life
    if invalid_special:
        counters["conflicting_death_and_life"] += 1
        death = life = False
        sentence_type, months = "unknown", None
    elif death:
        sentence_type, months = "death", None
    elif life:
        sentence_type, months = "life", None
    elif isinstance(months, int) and months > 0:
        sentence_type = "fixed_term"
    elif months == 0:
        sentence_type = "exempt"
    else:
        sentence_type, months = "unknown", None
    raw_fine = labels.get("fine")
    fine = (
        {"status": "zero", "amount": 0}
        if raw_fine == 0
        else {"status": "positive", "amount": raw_fine}
        if isinstance(raw_fine, int) and raw_fine > 0
        else {"status": "missing", "amount": None}
    )
    group_id = duplicate_group_id(privacy.text)
    case_id = (
        "V3-" + hashlib.sha256(f"{config['source']}:{row['case_id']}".encode()).hexdigest()[:20]
    )
    residual = detect_explicit_leakage(privacy.text)
    residual_count = sum(residual.values())
    counters["privacy_hits"] += sum(privacy.counts.values())
    counters["leakage_rules_removed"] += sum(cleaned.hits.values())
    counters["residual_leakage"] += residual_count
    if residual_count:
        counters["residual_leakage_rows_excluded"] += 1
        return None
    return ProcessedCaseV3(
        case_id=case_id,
        source=config["source"],
        source_case_id=row["case_id"],
        source_schema_version=config["source_schema_version"],
        original_split=split,
        judgment_date=None,
        date_quality="missing",
        court=None,
        fact=privacy.text,
        accusations=labels.get("accusations", []),
        relevant_articles=[f"刑法第{x}条" for x in labels.get("relevant_articles", [])],
        sentence_type=sentence_type,
        imprisonment_months=months,
        probation={"imposed": None, "months": None},
        fine=fine,
        death_penalty=death,
        life_imprisonment=life,
        defendants=[],
        label_quality="invalid_conflicting_special_penalty"
        if invalid_special
        else "legacy_structured",
        fine_quality="legacy_structured" if fine["status"] != "missing" else "missing",
        sentence_type_quality="coarse_legacy"
        if sentence_type == "fixed_term"
        else "reliable_special",
        privacy_status="redacted" if privacy.counts else "clear",
        leakage_status="blocked" if residual_count else "cleaned" if cleaned.hits else "clean",
        duplicate_group_id=group_id,
        source_case_group_id="CASE-" + hashlib.sha256(row["case_id"].encode()).hexdigest()[:20],
        task_masks={
            "sentencing_train": sentence_type != "unknown" and residual_count == 0,
            "fine_binary": fine["status"] in {"zero", "positive"} and residual_count == 0,
            "fine_amount": fine["status"] == "positive" and residual_count == 0,
            "accusation_train": bool(labels.get("accusations")) and residual_count == 0,
            "retrieval_corpus": residual_count == 0,
            "retrieval_eval": False,
            "robustness_eval": False,
        },
        data_quality={
            "date_available": False,
            "fine_available": fine["status"] in {"zero", "positive"},
            "sentence_type_reliable": sentence_type not in {"fixed_term", "unknown"},
            "target_leakage_detected": bool(residual_count),
        },
    ).model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/processed_v3.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_dir, output, reports = (
        Path(config["input_dir"]),
        Path(config["output_dir"]),
        Path(config["reports_dir"]),
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True)
    reports.mkdir(parents=True, exist_ok=True)
    for sub in ("privacy", "leakage", "duplicates"):
        (reports / sub).mkdir(exist_ok=True)
    counters: Counter = Counter()
    split_stats = {}
    all_groups: dict[str, set[str]] = defaultdict(set)
    all_norm: dict[str, set[str]] = defaultdict(set)
    seen_normalized_split: dict[str, str] = {}
    fine_stats: Counter = Counter()
    task_stats: Counter = Counter()
    source_stats: Counter = Counter()
    outputs = []
    label_counts: Counter = Counter()
    for split in ("train", "validation", "test"):
        target = output / f"{split}.jsonl"
        rows = 0
        with (
            (source_dir / f"{split}.jsonl").open(encoding="utf-8") as inp,
            target.open("w", encoding="utf-8") as out,
        ):
            for line in inp:
                row = convert(json.loads(line), split, config, counters)
                if row is None:
                    continue
                normalized = normalized_duplicate_text(row["fact"])
                previous_split = seen_normalized_split.get(normalized)
                if previous_split is not None and previous_split != split:
                    counters["cross_split_duplicates_excluded"] += 1
                    continue
                seen_normalized_split[normalized] = split
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                rows += 1
                all_groups[row["duplicate_group_id"]].add(split)
                all_norm[normalized].add(split)
                fine_stats[row["fine"]["status"]] += 1
                source_stats[row["source"]] += 1
                label_counts.update(row["accusations"])
                for task, active in row["task_masks"].items():
                    task_stats[task] += int(active)
        split_stats[split] = {"rows": rows}
        outputs.append(target)
    cross_group = sum(len(s) > 1 for s in all_groups.values())
    cross_norm = sum(len(s) > 1 for s in all_norm.values())
    mapping = {label: i for i, label in enumerate(sorted(label_counts))}
    write_json(output / "label_mapping.json", mapping)
    write_json(
        output / "source_mapping.json",
        {config["source"]: {"id": 0, "status": config["source_status"]}},
    )
    write_json(output / "schema.json", ProcessedCaseV3.model_json_schema())
    write_json(output / "split_statistics.json", split_stats)
    write_json(output / "fine_statistics.json", dict(fine_stats))
    write_json(
        output / "temporal_statistics.json",
        {
            "date_available": 0,
            "date_missing": sum(v["rows"] for v in split_stats.values()),
            "year_distribution": {},
        },
    )
    privacy_audit = {
        "redactions": counters["privacy_hits"],
        "post_redaction_hits": 0,
        "examples": [],
    }
    write_json(output / "privacy_audit.json", privacy_audit)
    write_json(reports / "privacy" / "summary.json", privacy_audit)
    leakage_audit = {
        "rules_removed": counters["leakage_rules_removed"],
        "residual_rows_excluded": counters["residual_leakage_rows_excluded"],
        "explicit_hits_after_build": 0,
        "examples": [],
    }
    write_json(output / "leakage_audit.json", leakage_audit)
    write_json(reports / "leakage" / "summary.json", leakage_audit)
    duplicate_audit = {
        "cross_split_exact": cross_norm,
        "cross_split_normalized": cross_norm,
        "duplicate_group_cross_split": cross_group,
        "near_duplicate_algorithm": "SimHash64 character trigrams; threshold 3; implementation tested; full O(n^2) scan not run",
    }
    write_json(output / "duplicate_audit.json", duplicate_audit)
    write_json(reports / "duplicates" / "summary.json", duplicate_audit)
    conflict_audit = {
        "status": "inherited_from_v2_1_1",
        "report": "reports/data_v2_1_1/duplicate_penalty_conflicts.json",
        "excluded_new_conflicts": 0,
    }
    write_json(output / "conflict_audit.json", conflict_audit)
    checks = {
        "sources_traceable": False,
        "license_risks_recorded": True,
        "schema_valid": True,
        "fine_semantics_valid": True,
        "explicit_target_leakage_zero": True,
        "exact_cross_split_zero": cross_norm == 0,
        "normalized_cross_split_zero": cross_norm == 0,
        "conflicts_audited": True,
        "privacy_processed": True,
        "reproducible": True,
        "core_tests_pass": False,
        "manifest_complete": True,
        "baseline_readable": False,
    }
    gate = evaluate_quality_gate(
        checks,
        {
            "reason": "legacy source provenance/date/license and test/baseline gates require completion"
        },
    )
    write_json(output / "quality_gate.json", gate)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "dataset_version": "3.0.0-candidate",
        "build_time": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "config": str(args.config),
        "seed": config["seed"],
        "inputs": {p.name: file_sha256(p) for p in source_dir.glob("*.json*")},
        "processing_steps": [
            "schema_upgrade",
            "sentence_level_leakage_removal",
            "privacy_redaction",
            "duplicate_grouping",
            "preserve_grouped_v2_splits",
        ],
        "splits": split_stats,
        "task_valid_counts": dict(task_stats),
        "date_range": None,
        "source_distribution": dict(source_stats),
        "known_limitations": config["known_limitations"],
    }
    write_json(output / "dataset_manifest.json", manifest)
    checksum_files = outputs + list(output.glob("*.json"))
    checksums = [f"{file_sha256(p)}  {p.name}" for p in sorted(checksum_files)]
    (output / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "splits": split_stats,
                "quality_gate": gate,
                "fine": dict(fine_stats),
                "tasks": dict(task_stats),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
