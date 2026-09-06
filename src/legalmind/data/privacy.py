from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_RULES = {
    "identity_card": re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    "mobile": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "bank_card": re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
    "birth_date": re.compile(r"(?:出生于?|生于)\s*\d{4}年\d{1,2}月(?:\d{1,2}日)?"),
    "plate": re.compile(r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9×＊*]{5}"),
    "organization_code": re.compile(r"(?<![A-Z0-9])[A-Z0-9]{8}-[A-Z0-9](?![A-Z0-9])"),
    "address": re.compile(r"(?:住址?|户籍(?:地|所在地))[:：]?[^，。；]{3,80}"),
    "name": re.compile(r"(?:被告人|被害人|证人)\s*([\u4e00-\u9fff]{2,4})(?=[，。,；;])"),
}


@dataclass(frozen=True)
class PrivacyResult:
    text: str
    counts: dict[str, int]


def redact_privacy(text: str) -> PrivacyResult:
    value = text
    counts: Counter[str] = Counter()
    for name, pattern in _RULES.items():
        replacement = f"[{name.upper()}_REDACTED]"
        if name == "name":
            value, count = pattern.subn(lambda m: m.group(0).replace(m.group(1), replacement), value)
        else:
            value, count = pattern.subn(replacement, value)
        counts[name] += count
    return PrivacyResult(value, dict(counts))


def scan_privacy(text: str) -> dict[str, int]:
    return {name: len(pattern.findall(text)) for name, pattern in _RULES.items()}
