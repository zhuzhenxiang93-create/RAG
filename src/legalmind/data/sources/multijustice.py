from __future__ import annotations

import ast
import csv
from typing import Iterable

from legalmind.data.sources.base import SourceAdapter


def _literal(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


class MultiJusticeAdapter(SourceAdapter):
    name = "MultiJustice_MPMCP"

    def records(self) -> Iterable[dict]:
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                yield {
                    "source_case_id": str(item.get("index", "")),
                    "judgment_date": item.get("date") or None,
                    "fact": item.get("fact") or "",
                    "accusations": _literal(item.get("accusation", "")),
                    "articles": _literal(item.get("relevant_article", "")),
                    "sentence_raw": _literal(item.get("imprisonment", "")),
                    "fine_raw": None,
                    "defendant_judgement": _literal(item.get("defendant_judgement", "")),
                }
