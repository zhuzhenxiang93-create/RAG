from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

from legalmind.schemas import CaseRecord

_SPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).replace("\u3000", " ")
    value = _CONTROL_RE.sub("", value)
    return _SPACE_RE.sub(" ", value).strip()


def clean_fact(text: str) -> str:
    """Backward-compatible alias for the canonical NFKC normalizer."""
    return normalize_text(text)


def normalized_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def normalize_record(
    raw: dict[str, Any],
    source_split: str | None = None,
    label_to_id: dict[str, int] | None = None,
) -> CaseRecord:
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    legacy_output = raw.get("output") if isinstance(raw.get("output"), dict) else {}
    nested_labels = raw.get("labels") if isinstance(raw.get("labels"), dict) else {}
    fact = clean_fact(
        raw.get("fact") or raw.get("input") or raw.get("text") or raw.get("content") or ""
    )
    accusations = [
        str(item).strip()
        for item in _as_list(
            raw.get("accusations")
            or raw.get("accusation")
            or nested_labels.get("accusations")
            or meta.get("accusation")
            or legacy_output.get("罪名")
        )
        if str(item).strip()
    ]
    articles = [
        int(item)
        for item in _as_list(
            raw.get("relevant_articles")
            or nested_labels.get("relevant_articles")
            or meta.get("relevant_articles")
        )
        if str(item).strip().lstrip("-").isdigit()
    ]
    supplied_ids = _as_list(raw.get("accusation_ids"))
    accusation_ids = [int(item) for item in supplied_ids if str(item).isdigit()]
    if not accusation_ids and label_to_id:
        accusation_ids = [label_to_id[label] for label in accusations if label in label_to_id]
    case_id = str(raw.get("case_id") or raw.get("id") or "").strip()
    if not case_id:
        digest = hashlib.sha1(fact.encode("utf-8")).hexdigest()[:16]
        case_id = f"{source_split or 'case'}-{digest}"
    penalty = raw.get("penalty") or meta.get("term_of_imprisonment")
    if penalty is None and nested_labels:
        penalty = {
            "death_penalty": nested_labels.get("death_penalty", False),
            "life_imprisonment": nested_labels.get("life_imprisonment", False),
            "imprisonment_months": nested_labels.get("imprisonment_months"),
        }
    if penalty is None and legacy_output:
        penalty = {
            "death_penalty": bool(legacy_output.get("是否死刑", False)),
            "life_imprisonment": bool(legacy_output.get("是否无期", False)),
            "imprisonment_months": legacy_output.get("有期徒刑"),
            "fine": legacy_output.get("罚金"),
        }
    return CaseRecord(
        case_id=case_id,
        fact=fact,
        accusations=sorted(set(accusations)),
        accusation_ids=sorted(set(accusation_ids)),
        relevant_articles=sorted(set(articles)),
        penalty=penalty if isinstance(penalty, dict) else None,
        source_split=source_split or raw.get("source_split"),
    )
