from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from legalmind.data.contracts_v3 import FineLabel, FineStatus

_CN = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
       "六": 6, "七": 7, "八": 8, "九": 9}
_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def chinese_integer(text: str) -> int:
    if text.isdigit():
        return int(text)
    total = section = number = 0
    for char in text:
        if char in _CN:
            number = _CN[char]
        elif char in _UNITS:
            unit = _UNITS[char]
            if unit == 10000:
                total += (section + number) * unit
                section = number = 0
            else:
                section += (number or 1) * unit
                number = 0
        else:
            raise ValueError(f"unsupported Chinese numeral: {char}")
    return total + section + number


def parse_rmb(value: object, *, missing_status: FineStatus = FineStatus.missing) -> FineLabel:
    if value is None or (isinstance(value, str) and not value.strip()):
        return FineLabel(status=missing_status, amount=None)
    if isinstance(value, bool):
        return FineLabel(status=FineStatus.invalid, amount=None)
    if isinstance(value, (int, float, Decimal)):
        try:
            amount = int(Decimal(str(value)))
        except (InvalidOperation, ValueError):
            return FineLabel(status=FineStatus.invalid, amount=None)
        if amount < 0:
            return FineLabel(status=FineStatus.invalid, amount=None)
        return FineLabel(status=FineStatus.zero if amount == 0 else FineStatus.positive, amount=amount)
    if not isinstance(value, str):
        return FineLabel(status=FineStatus.invalid, amount=None)
    text = str(value).strip().replace(",", "").replace("人民币", "").replace("罚金", "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?|[零〇一二两三四五六七八九十百千万]+)\s*(万元|万|元)?", text)
    if not match:
        return FineLabel(status=FineStatus.invalid, amount=None)
    token, unit = match.groups()
    try:
        number = Decimal(token) if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token) else Decimal(chinese_integer(token))
        amount = int(number * (10000 if unit in {"万", "万元"} else 1))
    except (InvalidOperation, ValueError):
        return FineLabel(status=FineStatus.invalid, amount=None)
    if amount < 0:
        return FineLabel(status=FineStatus.invalid, amount=None)
    return FineLabel(status=FineStatus.zero if amount == 0 else FineStatus.positive, amount=amount)
