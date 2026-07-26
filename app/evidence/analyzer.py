"""Evidence relevance, aggregation and deterministic conflict checks."""

import re
from typing import Dict, List, Set, Tuple

from app.retrieval.tokenizer import tokenize
from app.schemas.qa import ConflictItem, EvidenceItem
from app.schemas.search import SearchResponse


class EvidenceAnalyzer:
    """Turn retrieval hits into auditable evidence and conflict records."""

    _negative_markers = {
        "不",
        "不得",
        "禁止",
        "无需",
        "未",
        "not",
        "never",
        "prohibited",
        "forbidden",
    }
    _number = re.compile(
        r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(年|月|天|日|小时|分钟|%|元|万元|gb|mb)?",
        re.IGNORECASE,
    )
    _version = re.compile(
        r"(?:v(?:ersion)?\s*)?(\d+(?:\.\d+){1,3})|(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})",
        re.IGNORECASE,
    )

    def analyze(
        self, query: str, retrieval: SearchResponse
    ) -> Tuple[List[EvidenceItem], List[ConflictItem]]:
        query_tokens = self._meaningful_tokens(query)
        evidence: List[EvidenceItem] = []
        for index, hit in enumerate(retrieval.results, 1):
            source_text = hit.parent_content or hit.content
            relevance = self._relevance(query_tokens, source_text)
            relation = "support" if relevance >= 0.18 else "irrelevant"
            evidence.append(
                EvidenceItem(
                    evidence_id="E{}".format(index),
                    document_id=hit.document_id,
                    filename=hit.filename,
                    chunk_id=hit.chunk_id,
                    page=hit.page,
                    section=hit.section,
                    text=self._excerpt(source_text),
                    relation=relation,
                    relevance=relevance,
                )
            )

        conflicts = self._conflicts(query_tokens, evidence)
        conflicting_ids = {
            evidence_id for conflict in conflicts for evidence_id in conflict.evidence_ids
        }
        for item in evidence:
            if item.evidence_id in conflicting_ids:
                item.relation = "contradict"
        return evidence, conflicts

    def _conflicts(
        self, query_tokens: Set[str], evidence: List[EvidenceItem]
    ) -> List[ConflictItem]:
        conflicts: List[ConflictItem] = []
        relevant = [item for item in evidence if item.relation == "support"]
        conflict_index = 0
        for left_index, left in enumerate(relevant):
            for right in relevant[left_index + 1 :]:
                if left.document_id == right.document_id:
                    continue
                shared = self._shared_context(query_tokens, left.text, right.text)
                same_section = bool(
                    left.section and right.section and left.section == right.section
                )
                if shared < 0.2 and not same_section:
                    continue

                conflict_type = self._conflict_type(left.text, right.text)
                if conflict_type is None:
                    continue
                conflict_index += 1
                conflicts.append(
                    ConflictItem(
                        conflict_id="C{}".format(conflict_index),
                        evidence_ids=[left.evidence_id, right.evidence_id],
                        description="{} and {} contain conflicting {} evidence.".format(
                            left.filename, right.filename, conflict_type
                        ),
                        conflict_type=conflict_type,
                    )
                )
        return conflicts

    def _conflict_type(self, left: str, right: str):
        left_versions = set(self._version.findall(left.lower()))
        right_versions = set(self._version.findall(right.lower()))
        if left_versions and right_versions and left_versions.isdisjoint(right_versions):
            return "version"

        left_measurements = self._measurements(left)
        right_measurements = self._measurements(right)
        for unit in set(left_measurements) & set(right_measurements):
            if unit and left_measurements[unit].isdisjoint(right_measurements[unit]):
                return "numeric"

        left_negative = self._contains_negative(left)
        right_negative = self._contains_negative(right)
        if left_negative != right_negative:
            return "negation"
        return None

    @classmethod
    def _contains_negative(cls, text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in cls._negative_markers)

    @classmethod
    def _relevance(cls, query_tokens: Set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = cls._meaningful_tokens(text)
        shared = query_tokens & text_tokens
        overlap = len(shared) / len(query_tokens)
        if any(re.search(r"[a-z0-9@_-]", token, re.IGNORECASE) for token in shared):
            overlap = max(overlap, 0.4)
        return round(min(overlap, 1.0), 4)

    @classmethod
    def _measurements(cls, text: str) -> Dict[str, Set[str]]:
        measurements: Dict[str, Set[str]] = {}
        for value, unit in cls._number.findall(text.lower()):
            normalized_unit = unit.lower()
            measurements.setdefault(normalized_unit, set()).add(value)
        return measurements

    @classmethod
    def _shared_context(cls, query_tokens: Set[str], left: str, right: str) -> float:
        if not query_tokens:
            return 0.0
        left_tokens = cls._meaningful_tokens(left)
        right_tokens = cls._meaningful_tokens(right)
        return len(query_tokens & left_tokens & right_tokens) / len(query_tokens)

    @staticmethod
    def _meaningful_tokens(text: str) -> Set[str]:
        return {token for token in tokenize(text) if len(token) >= 2}

    @staticmethod
    def _excerpt(text: str, limit: int = 700) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"
