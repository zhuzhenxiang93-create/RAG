from __future__ import annotations

import json
from typing import Iterable

from legalmind.data.sources.base import SourceAdapter


class CailAdapter(SourceAdapter):
    name = "CAIL2018"

    def records(self) -> Iterable[dict]:
        with self.path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                item = json.loads(line)
                meta = item.get("meta", {})
                yield {
                    "source_case_id": str(item.get("id", index)),
                    "fact": item.get("fact") or "",
                    "accusations": meta.get("accusation") or [],
                    "articles": meta.get("relevant_articles") or [],
                    "sentence_raw": meta.get("term_of_imprisonment"),
                    "fine_raw": meta.get("punish_of_money") if "punish_of_money" in meta else None,
                }
