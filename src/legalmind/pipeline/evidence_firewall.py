from __future__ import annotations

from typing import Any

from legalmind.data.privacy import redact_privacy
from legalmind.schemas import SearchHit


def safe_evidence_payload(hit: SearchHit, text_limit: int = 360) -> dict[str, Any]:
    payload = hit.model_dump()
    if hit.evidence_type == "case":
        result = redact_privacy(hit.text)
        compact = " ".join(result.text.split())
        payload["text"] = compact[:text_limit] + ("…" if len(compact) > text_limit else "")
        payload["privacy_redaction_counts"] = result.counts
        payload["presentation"] = "deidentified_summary"
    else:
        payload["presentation"] = "official_statute_text"
    return payload


def evidence_firewall(
    citation_check: dict[str, Any],
    case_hits: list[SearchHit],
    statute_hits: list[SearchHit],
    rejected_statutes: list[dict[str, object]],
    as_of_date: str | None,
) -> dict[str, Any]:
    checks = {
        "citations_grounded": bool(citation_check.get("valid")),
        "case_evidence_present": bool(case_hits),
        "as_of_date_supplied": as_of_date is not None,
        "applicable_statute_present": bool(statute_hits),
        "statute_sources_verified": bool(statute_hits)
        and all(
            hit.source_status in {"human_verified_official", "verified_official"}
            for hit in statute_hits
        ),
    }
    blocking = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not blocking,
        "checks": checks,
        "blocking_reasons": blocking,
        "rejected_statutes": rejected_statutes,
        "requires_manual_review": bool(blocking),
    }
