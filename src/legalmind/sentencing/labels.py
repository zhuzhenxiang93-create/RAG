from __future__ import annotations

from dataclasses import dataclass

from legalmind.sentencing.schemas import SentenceType

IMPRISONMENT_BINS = (0, 6, 12, 24, 36, 60, 120, 180, 300, 10_000)
FINE_BINS = (0, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 10**12)


@dataclass(frozen=True)
class SentencingTargets:
    sentence_type: str
    imprisonment_months: int | None
    imprisonment_bucket: int | None
    fine_known: bool
    fine_imposed: bool | None
    fine_amount: int | None
    fine_bucket: int | None


def sentence_type_from_labels(labels: dict) -> SentenceType:
    explicit = labels.get("sentence_type")
    if isinstance(explicit, str) and explicit in {item.value for item in SentenceType}:
        return SentenceType(explicit)
    if bool(labels.get("death_penalty", False)):
        return SentenceType.death
    if bool(labels.get("life_imprisonment", False)):
        return SentenceType.life
    months = labels.get("imprisonment_months")
    if isinstance(months, int) and months > 0:
        # CAIL's legacy label does not distinguish fixed term, detention and control.
        return SentenceType.fixed_term
    if months == 0:
        return SentenceType.exempt
    return SentenceType.unknown


def bucket_index(value: int, edges: tuple[int, ...]) -> int:
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    return len(edges) - 2


def bucket_range(index: int, edges: tuple[int, ...]) -> tuple[int, int]:
    lower = edges[index]
    upper = edges[index + 1]
    return lower, upper if upper >= 10**11 else upper - 1


def targets_from_labels(labels: dict) -> SentencingTargets:
    sentence_type = sentence_type_from_labels(labels)
    months = labels.get("imprisonment_months")
    months = months if isinstance(months, int) and months >= 0 else None
    fine = labels.get("fine")
    if isinstance(fine, dict):
        status = fine.get("status")
        fine = fine.get("amount") if status in {"zero", "positive"} else None
    fine = fine if isinstance(fine, int) and fine >= 0 else None
    return SentencingTargets(
        sentence_type=sentence_type.value,
        imprisonment_months=months,
        imprisonment_bucket=(
            bucket_index(months, IMPRISONMENT_BINS)
            if sentence_type is SentenceType.fixed_term and months is not None
            else None
        ),
        fine_known=fine is not None,
        fine_imposed=(fine > 0 if fine is not None else None),
        fine_amount=(fine if fine is not None and fine > 0 else None),
        fine_bucket=(bucket_index(fine, FINE_BINS) if fine is not None and fine > 0 else None),
    )


def labels_from_row(row: dict) -> dict:
    """Return legacy nested labels or the v3 top-level task contract."""
    labels = row.get("labels")
    return labels if isinstance(labels, dict) else row
