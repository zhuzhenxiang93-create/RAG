from __future__ import annotations

from collections import Counter


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def split_statistics(records: list[dict], num_labels: int) -> dict:
    label_counts = Counter(
        label_id for record in records for label_id in record.get("accusation_ids", [])
    )
    lengths = [int(record["length_metadata"]["characters"]) for record in records]
    return {
        "rows": len(records),
        "single_label_rows": sum(len(row.get("accusation_ids", [])) == 1 for row in records),
        "multi_label_rows": sum(len(row.get("accusation_ids", [])) > 1 for row in records),
        "labels_present": len(label_counts),
        "labels_missing": sorted(set(range(num_labels)) - set(label_counts)),
        "label_frequency_min": min(label_counts.values(), default=0),
        "label_frequency_max": max(label_counts.values(), default=0),
        "fact_characters": {
            "min": min(lengths, default=0),
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths, default=0),
        },
        "target_leakage_removed_rows": sum(
            bool(row["cleaning_metadata"]["target_leakage_removed"]) for row in records
        ),
        "review_status": dict(Counter(row.get("review_status", "unreviewed") for row in records)),
    }


def frequency_bands(label_counts: Counter[str]) -> dict[str, dict[str, int]]:
    """Use distribution-derived quartiles while keeping thresholds interpretable."""
    counts = sorted(label_counts.values())
    if not counts:
        return {}
    q1 = counts[max(0, int(len(counts) * 0.25) - 1)]
    q3 = counts[max(0, int(len(counts) * 0.75) - 1)]
    return {
        "extreme_tail": {"min": 1, "max": q1},
        "tail": {"min": q1 + 1, "max": q3},
        "head": {"min": q3 + 1, "max": counts[-1]},
    }
