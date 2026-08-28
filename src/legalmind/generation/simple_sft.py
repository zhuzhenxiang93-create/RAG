from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from legalmind.data.sampling import select_multilabel_subset
from legalmind.generation.schemas import SimplifiedLegalAnalysis
from legalmind.generation.structured import extract_key_facts

SYSTEM_PROMPT_V3 = (
    "你是刑事案件算法实验助手。只输出一个合法JSON对象，不要输出解释或Markdown。"
    "字段名必须严格使用candidate_accusations、key_facts、confidence、"
    "requires_manual_review。confidence只能是high、medium、low。"
    "不得增加输入事实中不存在的信息。本输出不构成法律意见。"
)


def build_simple_target(
    row: dict, construction_type: str = "deterministic_simplified_schema"
) -> SimplifiedLegalAnalysis:
    if construction_type == "deterministic_insufficient_information":
        return SimplifiedLegalAnalysis(
            candidate_accusations=[],
            key_facts=extract_key_facts(row["fact"], limit=1),
            confidence="low",
            requires_manual_review=True,
        )
    accusations = list(row["labels"]["accusations"])
    requires_review = len(accusations) > 1 or len(row["fact"]) >= 1120
    return SimplifiedLegalAnalysis(
        candidate_accusations=accusations,
        key_facts=extract_key_facts(row["fact"], limit=3),
        confidence="low" if requires_review else "medium",
        requires_manual_review=requires_review,
    )


def build_simple_sft_record(
    row: dict,
    prompt_version: str = "legal-simple-json-v3",
    construction_type: str = "deterministic_simplified_schema",
) -> dict:
    input_fact = row["fact"]
    source_fact_sha256 = row.get("cleaning_metadata", {}).get("fact_sha256")
    if construction_type == "deterministic_insufficient_information":
        sentences = [
            value.strip() for value in re.split(r"(?<=[。！？；])", input_fact) if value.strip()
        ]
        input_fact = sentences[0] if sentences else input_fact[:160]
        row = {**row, "fact": input_fact}
    target = build_simple_target(row, construction_type)
    target_json = json.dumps(
        target.model_dump(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return {
        "sft_id": f"{row['case_id']}:{construction_type}",
        "case_id": row["case_id"],
        "source_split": row["source_split"],
        "construction_type": construction_type,
        "prompt_version": prompt_version,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_V3},
            {"role": "user", "content": f"案件事实：{input_fact}"},
            {"role": "assistant", "content": target_json},
        ],
        "target": target.model_dump(),
        "input_fact_sha256": hashlib.sha256(input_fact.encode("utf-8")).hexdigest(),
        "source_fact_sha256": source_fact_sha256,
        "automatic_validation": {
            "json_schema_valid": True,
            "labels_from_source": True,
            "contains_unverified_articles": False,
            "synthetic_abstention_target": construction_type
            == "deterministic_insufficient_information",
        },
        "review_status": "unreviewed",
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_simple_sft_file(
    source: Path,
    target: Path,
    sample_size: int,
    num_labels: int,
    seed: int,
    uncertainty_rows: int = 0,
) -> dict:
    rows = read_jsonl(source)
    sampled, report = select_multilabel_subset(rows, sample_size, num_labels, seed=seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sampled:
            handle.write(json.dumps(build_simple_sft_record(row), ensure_ascii=False) + "\n")
        for row in sampled[:uncertainty_rows]:
            record = build_simple_sft_record(
                row, construction_type="deterministic_insufficient_information"
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        **report,
        "output_rows": len(sampled) + min(uncertainty_rows, len(sampled)),
        "construction_type_distribution": {
            "deterministic_simplified_schema": len(sampled),
            "deterministic_insufficient_information": min(uncertainty_rows, len(sampled)),
        },
    }
