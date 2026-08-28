from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from legalmind.config import load_yaml
from legalmind.data.loader import iter_cases


def audit(path: Path, split: str, min_text_chars: int) -> dict:
    label_counts: Counter[str] = Counter()
    article_counts: Counter[int] = Counter()
    rows = empty_labels = multi_label = short_text = duplicate_text = 0
    fingerprints: set[int] = set()
    for case in iter_cases(path, split):
        rows += 1
        label_counts.update(case.accusations)
        article_counts.update(case.relevant_articles)
        empty_labels += int(not case.accusations)
        multi_label += int(len(case.accusations) > 1)
        short_text += int(len(case.fact) < min_text_chars)
        fingerprint = hash(case.fact)
        duplicate_text += int(fingerprint in fingerprints)
        fingerprints.add(fingerprint)
    return {
        "path": str(path),
        "rows": rows,
        "unique_labels": len(label_counts),
        "unique_articles": len(article_counts),
        "empty_label_rows": empty_labels,
        "multi_label_rows": multi_label,
        "short_text_rows": short_text,
        "duplicate_text_rows": duplicate_text,
        "top_labels": label_counts.most_common(20),
        "tail_labels": sorted(label_counts.items(), key=lambda item: item[1])[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default="artifacts/results/data_audit.json")
    args = parser.parse_args()
    config = load_yaml(args.config)
    report = {}
    for split, value in config["raw"].items():
        if value and Path(value).exists():
            report[split] = audit(Path(value), split, int(config.get("min_text_chars", 20)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
