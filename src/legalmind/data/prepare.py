from __future__ import annotations

import re

from legalmind.schemas import CaseRecord

_LABEL_BRACKETS = str.maketrans("", "", "[]【】")
_LOW_INFORMATION_PATTERNS = (
    re.compile(
        r"公诉机关指控的(?:罪名和)?(?:犯罪)?事实与.*?"
        r"(?:查明|认定)的(?:犯罪)?事实(?:基本)?(?:一致|相同)"
    ),
    re.compile(
        r"(?:经)?(?:二审|本院|审理).*?(?:事实|证据).*?"
        r"(?:一审|原审|原判).*?(?:一致|相同|确认|清楚)"
    ),
    re.compile(
        r"(?:经)?(?:二审|本院|审理).*?(?:一审|原审|原判).*?"
        r"(?:事实|证据).*?(?:一致|相同|确认|清楚)"
    ),
    re.compile(
        r"(?:经)?审理查明.*?(?:一审|原审|原判).*?"
        r"(?:事实|证据|认定).*?(?:一致|相同|确认|清楚)"
    ),
)


def canonicalize_label(label: str) -> str:
    return re.sub(r"\s+", "", label.translate(_LABEL_BRACKETS)).strip()


def canonicalize_labels(labels: list[str]) -> list[str]:
    return sorted({value for label in labels if (value := canonicalize_label(label))})


def is_low_information_fact(text: str, max_chars: int = 120) -> bool:
    compact = re.sub(r"\s+", "", text).strip()
    if len(compact) > max_chars:
        return False
    return any(pattern.search(compact) for pattern in _LOW_INFORMATION_PATTERNS)


def merge_case_labels(target: CaseRecord, incoming: CaseRecord) -> bool:
    """Merge duplicate annotations and report whether their label sets conflict."""
    target_labels = set(target.accusations)
    incoming_labels = set(incoming.accusations)
    conflict = target_labels != incoming_labels
    target.accusations = sorted(target_labels | incoming_labels)
    target.relevant_articles = sorted(
        set(target.relevant_articles) | set(incoming.relevant_articles)
    )
    return conflict
