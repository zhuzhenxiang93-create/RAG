from __future__ import annotations

import hashlib

from legalmind.data.privacy import redact_privacy
from legalmind.generation.contracts_v2 import EvidenceItemV1, EvidencePacketV1
from legalmind.pipeline.evidence_firewall import safe_evidence_payload
from legalmind.schemas import SearchHit


def _request_id(fact: str, as_of_date: str | None) -> str:
    digest = hashlib.sha256(f"{as_of_date or ''}\0{fact}".encode()).hexdigest()
    return f"req-{digest[:16]}"


def _item(hit: SearchHit) -> EvidenceItemV1:
    payload = safe_evidence_payload(hit)
    title = hit.case_id if hit.evidence_type == "case" else f"法条 {hit.case_id}"
    return EvidenceItemV1(
        evidence_id=hit.chunk_id,
        evidence_type="statute" if hit.evidence_type == "statute" else "case",
        title=title,
        summary=str(payload["text"]),
        source_url=hit.source_url,
        source_status=hit.source_status,
        effective_date=hit.effective_date,
        expiry_date=hit.expiry_date,
        legal_status=hit.legal_status,
    )


def build_evidence_packet(
    fact: str,
    predicted_accusations: list[str],
    case_hits: list[SearchHit],
    statute_hits: list[SearchHit],
    *,
    as_of_date: str | None,
    sentencing_baseline: dict[str, object] | None = None,
) -> EvidencePacketV1:
    safe_fact = redact_privacy(fact).text
    return EvidencePacketV1(
        request_id=_request_id(safe_fact, as_of_date),
        fact=safe_fact,
        as_of_date=as_of_date,
        predicted_accusations=predicted_accusations,
        sentencing_baseline=sentencing_baseline,
        evidence=[*map(_item, case_hits), *map(_item, statute_hits)],
    )
