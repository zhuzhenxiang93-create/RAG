from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from legalmind.generation.contracts_v2 import EvidencePacketV1, LegalAnalysisV1
from legalmind.generation.grounding_validator import validate_grounded_analysis


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = Counter()
    checksums = {}
    for split in ("train", "validation", "test"):
        path = args.dataset / f"{split}.jsonl"
        checksums[split] = sha256_file(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                counts["rows"] += 1
                counts[f"split_{split}"] += 1
                row = json.loads(line)
                packet = EvidencePacketV1.model_validate(row["evidence_packet"])
                target = LegalAnalysisV1.model_validate(row["target"])
                report = validate_grounded_analysis(
                    target, packet, allow_synthetic_test_evidence=True
                )
                counts["schema_valid"] += 1
                counts["grounding_valid"] += int(report["valid"])
                counts["unknown_citations"] += len(report["unknown_evidence_ids"])
                counts["statute_validity_errors"] += len(report["statute_validity_errors"])
                counts["privacy_hits"] += sum(report["privacy_hits"].values())
                counts[f"disposition_{target.disposition}"] += 1
                counts[f"review_{row['review_status']}"] += 1
    total = counts["rows"]
    result = {
        "dataset": str(args.dataset),
        "dataset_content": "synthetic_contract_test_only",
        "rows": total,
        "splits": {name: counts[f"split_{name}"] for name in ("train", "validation", "test")},
        "schema_valid_rate": counts["schema_valid"] / total,
        "grounding_valid_rate": counts["grounding_valid"] / total,
        "unknown_citation_count": counts["unknown_citations"],
        "statute_validity_error_count": counts["statute_validity_errors"],
        "privacy_hit_count": counts["privacy_hits"],
        "dispositions": {
            "analyzed": counts["disposition_analyzed"],
            "insufficient_evidence": counts["disposition_insufficient_evidence"],
        },
        "human_review": {
            "unreviewed": counts["review_unreviewed"],
            "completed": 0,
            "required_before_pilot_training": 200,
        },
        "checksums": checksums,
        "automatic_gate_passed": (
            counts["schema_valid"] == total
            and counts["grounding_valid"] == total
            and counts["unknown_citations"] == 0
            and counts["statute_validity_errors"] == 0
            and counts["privacy_hits"] == 0
        ),
        "pilot_training_gate_passed": False,
        "pilot_training_blocker": "200-item human legal review is incomplete",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
