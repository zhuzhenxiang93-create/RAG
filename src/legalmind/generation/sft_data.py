from __future__ import annotations

import hashlib
import json
from pathlib import Path

from legalmind.generation.structured import deterministic_grounded_analysis

SYSTEM_PROMPT = (
    "你是刑事案件算法实验助手。只能根据输入事实和给出的检索证据输出JSON；"
    "证据不足时列出缺失信息并要求人工复核。本输出不构成法律意见。"
)


def build_instruction(row: dict) -> str:
    return (
        f"案件事实：{row['fact']}\n"
        "已验证法条：[]\n相似案例：[]\n"
        "请输出罪名候选、关键事实、缺失信息、置信度与人工复核标记。"
    )


def build_sft_record(row: dict, prompt_version: str = "legal-sft-v1") -> dict:
    accusations = list(row["labels"]["accusations"])
    articles = list(row["labels"].get("relevant_articles", []))
    target = deterministic_grounded_analysis(row["fact"], accusations, [], articles)
    return {
        "case_id": row["case_id"],
        "source_split": row["source_split"],
        "construction_type": "deterministic_from_cail_labels",
        "prompt_version": prompt_version,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_instruction(row)},
            {"role": "assistant", "content": json.dumps(target.model_dump(), ensure_ascii=False)},
        ],
        "target": target.model_dump(),
        "input_fact_sha256": hashlib.sha256(row["fact"].encode("utf-8")).hexdigest(),
        "automatic_validation": {
            "json_schema_valid": True,
            "labels_from_source": True,
            "citations_grounded": not articles,
            "citation_limitation": "legacy source has no verified article field"
            if not articles
            else None,
        },
        "review_status": "unreviewed",
    }


def build_sft_file(source: Path, target: Path, max_rows: int | None = None) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        source.open(encoding="utf-8") as source_handle,
        target.open("w", encoding="utf-8", newline="\n") as target_handle,
    ):
        for line in source_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            target_handle.write(json.dumps(build_sft_record(row), ensure_ascii=False) + "\n")
            count += 1
            if max_rows and count >= max_rows:
                break
    return count
