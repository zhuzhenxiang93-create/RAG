from __future__ import annotations

import json
import zipfile
from typing import Iterable

from legalmind.data.sources.base import SourceAdapter


class HrnAdapter(SourceAdapter):
    name = "MultiLJP_HRN"

    def records(self) -> Iterable[dict]:
        with zipfile.ZipFile(self.path) as archive:
            for name in archive.namelist():
                if not name.endswith((".json", ".jsonl")):
                    continue
                for line in archive.read(name).decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    for defendant, labels in (item.get("criminals_info") or {}).items():
                        yield {
                            "source_case_id": str(item.get("id", name)),
                            "defendant_id": defendant,
                            "fact": item.get("fact") or "",
                            "accusations": labels.get("accusations") or [],
                            "articles": labels.get("laws") or [],
                            "sentence_raw": labels.get("term"),
                            "fine_raw": None,
                            "alignment_reliable": defendant in (item.get("fact") or ""),
                        }
