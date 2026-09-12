from __future__ import annotations

import re
from dataclasses import dataclass


_GENERIC_PATTERNS = (
    ("explicit_charge_conclusion", re.compile(r"(?:其行为|上述行为|被告人行为)?(?:已)?构成[^，。；]{1,30}?罪")),
    ("explicit_prosecution_charge", re.compile(r"应当?以[^，。；]{1,30}?罪(?:追究|定罪|处罚)")),
    ("explicit_charge_request", re.compile(r"以[^，。；]{1,30}?罪(?:提起公诉|判处)")),
    (
        "explicit_verdict_charge",
        re.compile(r"(?:判决如下[:：]?\s*)?被告人[^，。；]{0,24}?犯[^，。；]{1,30}?罪"),
    ),
    (
        "explicit_death_or_life_sentence",
        re.compile(r"(?:决定执行|判处)[^，。；]{0,30}?(?:死刑|无期徒刑)"),
    ),
    (
        "explicit_term_sentence",
        re.compile(r"(?:决定执行|判处)[^，。；]{0,30}?(?:有期徒刑|拘役|管制)[^，。；]{0,24}"),
    ),
    (
        "generic_sentence_after_verdict_verb",
        re.compile(r"(?:判处|决定执行|处以)[^，。；]{1,100}"),
    ),
    (
        "explicit_probation_sentence",
        re.compile(r"(?:宣告)?缓刑[一二三四五六七八九十百零〇0-9年月个]+"),
    ),
    (
        "explicit_fine_sentence",
        re.compile(r"(?:并处罚金|判处罚金|处罚金|并处)[^，。；]{0,40}"),
    ),
)


@dataclass(frozen=True)
class LeakageResult:
    text: str
    removed: bool
    leakage_types: tuple[str, ...]


def strip_target_leakage(text: str) -> LeakageResult:
    value = text
    found: list[str] = []
    for name, pattern in _GENERIC_PATTERNS:
        value, count = pattern.subn("[罪名结论已脱敏]", value)
        if count:
            found.append(name)
    value = re.sub(r"(?:\[罪名结论已脱敏\]\s*){2,}", "[罪名结论已脱敏]", value)
    return LeakageResult(value.strip(), bool(found), tuple(found))


def detect_article_leakage(text: str) -> list[int]:
    values = re.findall(r"《中华人民共和国刑法》第([一二三四五六七八九十百千零〇0-9]+)条", text)
    return [int(value) for value in values if value.isdigit()]
