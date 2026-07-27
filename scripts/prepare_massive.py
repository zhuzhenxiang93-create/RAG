"""Download and normalize MASSIVE for reproducible intent experiments."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict


def _name(feature, value) -> str:
    return feature.int2str(value) if hasattr(feature, "int2str") else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locale", action="append", default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default="data/intent/massive")
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install Full dependencies first: pip install -r requirements-full.txt"
        ) from exc

    locales = args.locale or ["zh-CN"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    taxonomy: Dict[str, str] = {}
    split_counts: Dict[str, int] = {}
    digest = hashlib.sha256()

    for locale in locales:
        dataset = load_dataset("AmazonScience/massive", locale, revision=args.revision)
        for split_name, split in dataset.items():
            intent_feature = split.features["intent"]
            scenario_feature = split.features["scenario"]
            target = output / f"{locale}.{split_name}.jsonl"
            count = 0
            with target.open("w", encoding="utf-8", newline="\n") as handle:
                for row in split:
                    if args.limit_per_split is not None and count >= args.limit_per_split:
                        break
                    intent = _name(intent_feature, row["intent"])
                    domain = _name(scenario_feature, row["scenario"])
                    taxonomy[intent] = domain
                    normalized = {
                        "id": str(row.get("id", f"{locale}-{split_name}-{count}")),
                        "locale": locale,
                        "text": row.get("utt") or row.get("text"),
                        "intent": intent,
                        "domain": domain,
                    }
                    line = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
                    handle.write(line + "\n")
                    digest.update((line + "\n").encode("utf-8"))
                    count += 1
            split_counts[f"{locale}.{split_name}"] = count

    intents = {label: index for index, label in enumerate(sorted(taxonomy))}
    taxonomy_payload = {
        "dataset": "AmazonScience/massive",
        "revision": args.revision,
        "locales": locales,
        "intents": intents,
        "intent_to_domain": {label: taxonomy[label] for label in sorted(taxonomy)},
    }
    (output / "taxonomy.json").write_text(
        json.dumps(taxonomy_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "dataset": "AmazonScience/massive",
        "revision": args.revision,
        "locales": locales,
        "split_counts": split_counts,
        "normalized_sha256": digest.hexdigest(),
        "limited": args.limit_per_split is not None,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

