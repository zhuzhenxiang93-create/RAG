from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from legalmind.generation.schemas import SimplifiedLegalAnalysis


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_sft(data_dir: Path) -> dict:
    split_rows = {
        "train": read_jsonl(data_dir / "train.jsonl"),
        "validation": read_jsonl(data_dir / "validation.jsonl"),
    }
    errors: Counter[str] = Counter()
    construction: Counter[str] = Counter()
    split_cases: dict[str, set[str]] = {}
    split_ids: dict[str, set[str]] = {}
    for split, rows in split_rows.items():
        split_cases[split] = set()
        split_ids[split] = set()
        for row in rows:
            construction[row.get("construction_type", "missing")] += 1
            split_cases[split].add(row["case_id"])
            sft_id = row.get("sft_id")
            if not sft_id or sft_id in split_ids[split]:
                errors["missing_or_duplicate_sft_id"] += 1
            split_ids[split].add(sft_id)
            if row.get("source_split") != split:
                errors["source_split_mismatch"] += 1
            if row.get("review_status") != "unreviewed":
                errors["invalid_review_status"] += 1
            try:
                target = SimplifiedLegalAnalysis.model_validate(row["target"])
            except (ValueError, TypeError):
                errors["target_schema_invalid"] += 1
                continue
            try:
                assistant = SimplifiedLegalAnalysis.model_validate_json(
                    row["messages"][-1]["content"]
                )
            except (ValueError, TypeError):
                errors["assistant_json_invalid"] += 1
                continue
            if assistant != target:
                errors["assistant_target_mismatch"] += 1
            if row["construction_type"] == "deterministic_insufficient_information":
                if target.candidate_accusations:
                    errors["insufficient_information_has_accusation"] += 1
                if target.confidence != "low" or not target.requires_manual_review:
                    errors["insufficient_information_not_escalated"] += 1
    overlap = split_cases["train"].intersection(split_cases["validation"])
    if overlap:
        errors["source_case_cross_split_overlap"] += len(overlap)
    return {
        "rows": {split: len(rows) for split, rows in split_rows.items()},
        "construction_type_distribution": dict(construction),
        "source_case_cross_split_overlap": len(overlap),
        "errors": dict(errors),
        "valid": not errors,
        "test_rows_used": 0,
        "review_status": "unreviewed",
    }


def validate_retrieval(data_dir: Path) -> dict:
    queries = read_jsonl(data_dir / "queries.jsonl")
    candidates = read_jsonl(data_dir / "candidate_pool.jsonl")
    qrels = read_jsonl(data_dir / "silver_qrels.jsonl")
    reviewed = read_jsonl(data_dir / "reviewed_qrels.jsonl")
    query_ids = {row["query_id"] for row in queries}
    candidate_ids = {row["candidate_id"] for row in candidates}
    errors: Counter[str] = Counter()
    for row in queries:
        if row.get("source_split") != "validation":
            errors["query_not_validation"] += 1
        if row.get("review_status") != "unreviewed":
            errors["query_review_status_invalid"] += 1
    for row in candidates:
        if row.get("source_split") != "train":
            errors["candidate_not_train"] += 1
    for row in qrels:
        if row["query_id"] not in query_ids or row["candidate_id"] not in candidate_ids:
            errors["dangling_qrel"] += 1
        if row.get("judgment_source") != "silver_proxy":
            errors["qrel_claims_non_proxy_judgment"] += 1
        if row.get("review_status") != "unreviewed":
            errors["qrel_review_status_invalid"] += 1
    if reviewed:
        errors["reviewed_qrels_must_be_empty_before_human_review"] += len(reviewed)
    return {
        "queries": len(queries),
        "candidate_pool": len(candidates),
        "silver_qrels": len(qrels),
        "reviewed_qrels": len(reviewed),
        "query_candidate_id_overlap": len(query_ids.intersection(candidate_ids)),
        "errors": dict(errors),
        "valid": not errors,
        "judgment_status": "silver_proxy_unreviewed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-dir", default="data/sft_v3_1")
    parser.add_argument("--retrieval-dir", default="data/retrieval_eval")
    parser.add_argument("--output", default="reports/data/interview_dataset_validation.json")
    args = parser.parse_args()
    report = {
        "sft": validate_sft(Path(args.sft_dir)),
        "retrieval": validate_retrieval(Path(args.retrieval_dir)),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["sft"]["valid"] or not report["retrieval"]["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
