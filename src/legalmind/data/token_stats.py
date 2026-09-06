from __future__ import annotations


def summarize_token_lengths(
    lengths: list[int],
    thresholds: tuple[int, ...] = (256, 512, 1024, 2048, 3072, 4096),
) -> dict:
    ordered = sorted(lengths)
    if not ordered:
        return {"rows": 0}

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]

    summary = {
        "rows": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "thresholds": {},
    }
    for threshold in thresholds:
        count = sum(length > threshold for length in ordered)
        summary["thresholds"][str(threshold)] = {
            "rows_exceeding": count,
            "fraction_exceeding": count / len(ordered),
        }
    return summary
