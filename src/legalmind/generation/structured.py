from __future__ import annotations

import json
import re

from legalmind.generation.schemas import SimplifiedLegalAnalysis, StructuredLegalAnalysis
from legalmind.schemas import SearchHit


def extract_key_facts(fact: str, limit: int = 4) -> list[str]:
    sentences = [value.strip() for value in re.split(r"(?<=[。！？；])", fact) if value.strip()]
    return sentences[:limit]


def deterministic_grounded_analysis(
    fact: str,
    predicted_accusations: list[str],
    hits: list[SearchHit],
    relevant_articles: list[int] | None = None,
) -> StructuredLegalAnalysis:
    articles = sorted(set(relevant_articles or []))
    cited_cases = [
        {"case_id": hit.case_id, "reason": "检索结果显示案件事实具有相似行为或结果"}
        for hit in hits[:5]
    ]
    missing = []
    if not articles:
        missing.append("缺少经过验证的适用法条")
    if not hits:
        missing.append("没有达到阈值的相似案例")
    confidence = "low" if missing or not predicted_accusations else "medium"
    return StructuredLegalAnalysis(
        predicted_accusations=predicted_accusations,
        relevant_articles=articles,
        key_facts=extract_key_facts(fact),
        missing_information=missing,
        similar_cases=cited_cases,
        analysis="结论仅基于输入事实与当前可追溯检索证据，仍需专业人员复核。",
        confidence=confidence,
        requires_manual_review=confidence != "high",
    )


def parse_and_validate_json(value: str) -> StructuredLegalAnalysis:
    return StructuredLegalAnalysis.model_validate(json.loads(extract_json_object(value)))


def extract_json_object(value: str) -> str:
    text = value.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("JSON object is truncated")


_SIMPLIFIED_KEYS = {
    "candidate_accusations": "candidate_accusations",
    "predicted_accusations": "candidate_accusations",
    "罪名候选": "candidate_accusations",
    "预测罪名": "candidate_accusations",
    "key_facts": "key_facts",
    "关键事实": "key_facts",
    "confidence": "confidence",
    "置信度": "confidence",
    "requires_manual_review": "requires_manual_review",
    "人工复核标记": "requires_manual_review",
    "是否需要人工复核": "requires_manual_review",
}
_CONFIDENCE_VALUES = {
    "高": "high",
    "高置信度": "high",
    "high": "high",
    "中": "medium",
    "中等": "medium",
    "中置信度": "medium",
    "medium": "medium",
    "低": "low",
    "低置信度": "low",
    "low": "low",
}


def canonicalize_simplified_output(value: dict) -> dict:
    normalized = {}
    for key, item in value.items():
        mapped = _SIMPLIFIED_KEYS.get(key)
        if mapped:
            normalized[mapped] = item
    accusations = normalized.get("candidate_accusations", [])
    if isinstance(accusations, str):
        accusations = [accusations]
    normalized["candidate_accusations"] = [
        accusation.removesuffix("罪")
        for accusation in accusations
        if isinstance(accusation, str) and accusation.strip()
    ]
    facts = normalized.get("key_facts", [])
    normalized["key_facts"] = [facts] if isinstance(facts, str) else facts
    confidence = str(normalized.get("confidence", "low")).strip().lower()
    normalized["confidence"] = _CONFIDENCE_VALUES.get(confidence, "low")
    manual = normalized.get("requires_manual_review", True)
    if isinstance(manual, str):
        manual = manual.strip().lower() not in {"false", "否", "无需人工复核", "不需要"}
    normalized["requires_manual_review"] = bool(manual)
    return normalized


def parse_simplified_json(value: str, repair: bool = False) -> SimplifiedLegalAnalysis:
    parsed = json.loads(extract_json_object(value))
    if repair:
        parsed = canonicalize_simplified_output(parsed)
    return SimplifiedLegalAnalysis.model_validate(parsed)


def validate_citations(
    analysis: StructuredLegalAnalysis,
    retrieved_case_ids: set[str],
    retrieved_articles: set[int],
) -> dict[str, list[str] | bool]:
    invalid_cases = [
        row.case_id for row in analysis.similar_cases if row.case_id not in retrieved_case_ids
    ]
    invalid_articles = [
        str(article) for article in analysis.relevant_articles if article not in retrieved_articles
    ]
    return {
        "valid": not invalid_cases and not invalid_articles,
        "invalid_case_ids": invalid_cases,
        "invalid_articles": invalid_articles,
    }
