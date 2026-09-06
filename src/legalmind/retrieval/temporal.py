from __future__ import annotations

from datetime import date
from urllib.parse import urlparse

from legalmind.schemas import SearchHit

OFFICIAL_VERIFIED_STATUSES = {"human_verified_official", "verified_official"}
EFFECTIVE_STATUSES = {"effective", "现行有效"}
AUTHORITATIVE_DOMAIN_SUFFIXES = ("gov.cn", "court.gov.cn", "spp.gov.cn", "npc.gov.cn")


def is_authoritative_source(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    return parsed.scheme == "https" and any(
        host == suffix or host.endswith(f".{suffix}") for suffix in AUTHORITATIVE_DOMAIN_SUFFIXES
    )


def parse_iso_date(value: str | None, field: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must use YYYY-MM-DD: {value}") from error


def assess_statute_hit(hit: SearchHit, as_of_date: str | None) -> list[str]:
    """Return reasons why a statute hit is unsafe to present as applicable law."""
    reasons: list[str] = []
    if hit.evidence_type != "statute":
        reasons.append("not_statute_evidence")
    if not is_authoritative_source(hit.source_url):
        reasons.append("non_authoritative_source_url")
    if hit.source_status not in OFFICIAL_VERIFIED_STATUSES:
        reasons.append("source_not_human_verified")
    if hit.legal_status not in EFFECTIVE_STATUSES:
        reasons.append("legal_status_not_effective")
    if as_of_date is None:
        reasons.append("missing_as_of_date")
        return reasons
    target = parse_iso_date(as_of_date, "as_of_date")
    effective = parse_iso_date(hit.effective_date, "effective_date")
    expiry = parse_iso_date(hit.expiry_date, "expiry_date")
    if effective is None:
        reasons.append("missing_effective_date")
    elif target < effective:
        reasons.append("not_yet_effective")
    if expiry is not None and target >= expiry:
        reasons.append("expired")
    return reasons


def filter_effective_statutes(
    hits: list[SearchHit], as_of_date: str | None
) -> tuple[list[SearchHit], list[dict[str, object]]]:
    accepted: list[SearchHit] = []
    rejected: list[dict[str, object]] = []
    for hit in hits:
        reasons = assess_statute_hit(hit, as_of_date)
        if reasons:
            rejected.append({"clause_id": hit.chunk_id, "reasons": reasons})
        else:
            accepted.append(hit)
    return accepted, rejected
