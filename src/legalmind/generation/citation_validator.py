from __future__ import annotations

import re

from legalmind.schemas import SearchHit

_CITATION_RE = re.compile(r"\[(case-[^\]]+|[^:\]]+:c\d+)\]")


def validate_citations(text: str, evidence: list[SearchHit]) -> dict[str, list[str] | bool]:
    cited = set(_CITATION_RE.findall(text))
    allowed = {hit.chunk_id for hit in evidence}
    unknown = sorted(cited - allowed)
    return {
        "valid": bool(cited) and not unknown,
        "cited": sorted(cited),
        "unknown": unknown,
        "uncited_evidence": sorted(allowed - cited),
    }
