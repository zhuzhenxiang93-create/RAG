from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from legalmind.data.dedup_v3 import duplicate_group_id
from legalmind.data.leakage_v3 import detect_explicit_leakage
from legalmind.data.money import parse_rmb
from legalmind.data.privacy import scan_privacy
from legalmind.data.sources import CailAdapter, HrnAdapter, JudgeAdapter, MultiJusticeAdapter

ADAPTERS = {
    "judge": JudgeAdapter,
    "hrn": HrnAdapter,
    "multijustice": MultiJusticeAdapter,
    "cail": CailAdapter,
}
DATE_RE = re.compile(r"(?:（|\()?(20\d{2})(?:）|\))?[^\n]{0,30}?(?:刑初|年\d{1,2}月\d{1,2}日)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(source: str, path: Path, metadata: dict) -> dict:
    counters: Counter[str] = Counter()
    charges: Counter[str] = Counter()
    years: Counter[str] = Counter()
    sentence_types: Counter[str] = Counter()
    privacy: Counter[str] = Counter()
    leakage: Counter[str] = Counter()
    fine_amounts: list[int] = []
    fingerprints: Counter[str] = Counter()
    fields: set[str] = set()
    lengths: list[int] = []
    damaged = 0
    adapter = ADAPTERS[source](path)
    for record in adapter.records():
        try:
            counters["total_records"] += 1
            fields.update(record)
            fact = str(record.get("fact") or "")
            if not fact:
                counters["empty_fact"] += 1
                continue
            counters["parseable_records"] += 1
            lengths.append(len(fact))
            fingerprints[duplicate_group_id(fact)] += 1
            charges.update(str(x) for x in record.get("accusations", []) if str(x))
            privacy.update(scan_privacy(fact))
            leakage.update(detect_explicit_leakage(fact))
            date_value = record.get("judgment_date")
            year = str(date_value)[:4] if date_value else None
            if not year or not year.isdigit():
                match = DATE_RE.search(str(record.get("full_document") or ""))
                year = match.group(1) if match else None
            years[year or "missing"] += 1
            raw_sentence = str(record.get("sentence_raw") or "")
            kind = (
                "death"
                if "死刑" in raw_sentence
                else "life"
                if "无期" in raw_sentence
                else "detention"
                if "拘役" in raw_sentence
                else "control"
                if "管制" in raw_sentence
                else "fixed_term"
                if "有期徒刑" in raw_sentence or raw_sentence.isdigit()
                else "unknown"
            )
            sentence_types[kind] += 1
            fine = parse_rmb(record.get("fine_raw"))
            counters[f"fine_{fine.status.value}"] += 1
            if fine.amount and fine.status.value == "positive":
                fine_amounts.append(fine.amount)
            defendants = record.get("defendant_judgement") or []
            defendant_count = record.get("defendant_count") or (
                len(defendants) if isinstance(defendants, (list, dict)) else 0
            )
            counters["multi_defendant"] += int(defendant_count > 1)
            counters["multi_accusation"] += int(len(record.get("accusations", [])) > 1)
        except (KeyError, TypeError, ValueError):
            damaged += 1
    ordered_fines = sorted(fine_amounts)
    ordered_lengths = sorted(lengths)
    pct = lambda values, q: (
        values[min(len(values) - 1, int((len(values) - 1) * q))] if values else None
    )
    nonmissing_years = sorted(int(y) for y in years if y != "missing")
    result = {
        **metadata,
        "download_date": datetime.now(timezone.utc).date().isoformat(),
        "file": {
            "name": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "compressed_bytes": path.stat().st_size,
        },
        "total_records": counters["total_records"],
        "parseable_records": counters["parseable_records"],
        "damaged_records": damaged,
        "fields": sorted(fields),
        "judgment_date": {
            "min_year": min(nonmissing_years, default=None),
            "max_year": max(nonmissing_years, default=None),
            "by_year": dict(sorted(years.items())),
            "missing": years["missing"],
            "missing_rate": years["missing"] / max(1, counters["total_records"]),
        },
        "multi_defendant_records": counters["multi_defendant"],
        "multi_accusation_records": counters["multi_accusation"],
        "accusation_frequency": dict(charges.most_common()),
        "sentence_type_distribution": dict(sentence_types),
        "imprisonment_coverage": 1 - sentence_types["unknown"] / max(1, counters["total_records"]),
        "probation_coverage": None,
        "fine": {
            "coverage": (counters["fine_zero"] + counters["fine_positive"])
            / max(1, counters["total_records"]),
            "missing": counters["fine_missing"],
            "zero": counters["fine_zero"],
            "positive": counters["fine_positive"],
            "invalid": counters["fine_invalid"],
            "positive_quantiles": {
                "p50": pct(ordered_fines, 0.5),
                "p90": pct(ordered_fines, 0.9),
                "p99": pct(ordered_fines, 0.99),
                "max": max(ordered_fines, default=None),
            },
        },
        "text_length": {
            "p50": pct(ordered_lengths, 0.5),
            "p90": pct(ordered_lengths, 0.9),
            "p99": pct(ordered_lengths, 0.99),
            "max": max(ordered_lengths, default=None),
        },
        "exact_duplicates": None,
        "normalized_duplicates": sum(v - 1 for v in fingerprints.values() if v > 1),
        "same_fact_conflicting_labels": None,
        "target_leakage_hits": dict(leakage),
        "target_leakage_records_upper_bound": sum(leakage.values()),
        "privacy_hits": dict(privacy),
        "recommended_for_training": False,
        "recommended_use": metadata.get("recommended_use", "audit_only"),
        "limitations": metadata.get("limitations", []),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=ADAPTERS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    report = audit(args.source, args.input, metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = metadata["slug"] + "_audit"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {report['official_name']} 审计",
        "",
        f"- 真实审计记录：{report['total_records']}",
        f"- 可解析：{report['parseable_records']}；损坏：{report['damaged_records']}",
        f"- 年份：{report['judgment_date']['min_year']}–{report['judgment_date']['max_year']}；缺失 {report['judgment_date']['missing']}",
        f"- 罚金：missing={report['fine']['missing']}, zero={report['fine']['zero']}, positive={report['fine']['positive']}, invalid={report['fine']['invalid']}",
        f"- 用途：{report['recommended_use']}；进入训练：否",
        "",
        "报告仅包含聚合统计和哈希，不包含案件原文。",
    ]
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
