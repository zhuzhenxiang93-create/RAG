from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_BOUNDARY = re.compile(r"(?:法院认为|本院认为|判决如下|本院判决)[:：]?")
_RULES = {
    "charge_conclusion": re.compile(r"(?:犯|构成|以)[^，。；]{1,30}?罪(?:追究刑事责任)?"),
    "term": re.compile(r"(?:判处|决定执行)[^，。；]{0,30}?(?:有期徒刑|拘役|管制|无期徒刑|死刑)[^，。；]{0,25}"),
    "probation": re.compile(r"缓刑[一二三四五六七八九十百零〇0-9年月个]+"),
    "fine": re.compile(r"(?:并处|判处)?罚金(?:人民币)?[^，。；]{0,30}"),
}


@dataclass(frozen=True)
class LeakageCleanResult:
    text: str
    hits: dict[str, int]
    truncated: bool


def detect_explicit_leakage(text: str) -> dict[str, int]:
    hits = {name: len(pattern.findall(text)) for name, pattern in _RULES.items()}
    hits["judgment_boundary"] = len(_BOUNDARY.findall(text))
    return hits


def clean_input_fact(text: str) -> LeakageCleanResult:
    value = text.strip()
    hits: Counter[str] = Counter()
    boundary = _BOUNDARY.search(value)
    truncated = boundary is not None
    if boundary:
        hits["judgment_boundary"] += 1
        value = value[: boundary.start()].rstrip("，。；:： ")
    sentences = re.split(r"(?<=[。；;!?！？])", value)
    kept: list[str] = []
    for sentence in sentences:
        sentence_hits = {name: len(pattern.findall(sentence)) for name, pattern in _RULES.items()}
        if any(sentence_hits.values()):
            hits.update(sentence_hits)
        else:
            kept.append(sentence)
    return LeakageCleanResult("".join(kept).strip(), dict(hits), truncated)
