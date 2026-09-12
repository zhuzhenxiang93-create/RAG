from __future__ import annotations

import json
from typing import Iterable

from legalmind.data.sources.base import SourceAdapter


class JudgeAdapter(SourceAdapter):
    name = "JuDGE"

    def records(self) -> Iterable[dict]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for item in raw:
            fines = item.get("Fine")
            if isinstance(fines, list):
                fine_raw = fines[0] if len(fines) == 1 else None if not fines else {"unaligned": fines}
            else:
                fine_raw = fines
            sentences = item.get("Sentence") or []
            yield {
                "source_case_id": str(item.get("CaseId", "")),
                "fact": item.get("Fact") or "",
                "full_document": item.get("Full Document") or "",
                "judgment": item.get("Judgment") or "",
                "sentence_raw": "；".join(str(value) for value in sentences),
                "fine_raw": fine_raw,
                "defendant_count": len(sentences),
                "accusations": item.get("Crime Type") or [],
                "articles": item.get("Law Articles") or [],
            }
