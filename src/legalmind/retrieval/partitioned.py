from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

from legalmind.data.money import parse_rmb
from legalmind.retrieval.lexical import LexicalBM25Index
from legalmind.schemas import CaseChunk, LabelScore, SearchHit

_GENERIC_TERMS = re.compile(
    r"被告人|行为人|当事人|案发后|主动|已经|取得谅解|谅解|退赃|退赔|返还|人民币|财物"
)
_AMOUNT = re.compile(
    r"(?:价值|金额|价款|损失|共计|合计|人民币)\s*"
    r"([0-9]+(?:\.[0-9]+)?|[零〇一二两三四五六七八九十百千万]+)\s*(万元|万|元)"
)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})年")
_FACTORS = {
    "自首": re.compile(r"自首|主动投案"),
    "坦白": re.compile(r"坦白|如实供述"),
    "退赃退赔": re.compile(r"退赃|退赔|返还"),
    "取得谅解": re.compile(r"谅解"),
    "累犯或前科": re.compile(r"累犯|前科|曾因.*(?:判刑|刑事处罚)"),
    "未遂": re.compile(r"未遂"),
}


def canonical_accusation(value: str) -> str:
    return value.strip().removesuffix("罪")


def _partition_name(accusation: str) -> str:
    value = re.sub(r"[\\/\x00]", "_", accusation.strip()).strip(".")
    return value or "unknown"


def _normalized_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _query_text(text: str) -> str:
    reduced = _GENERIC_TERMS.sub("", text)
    return reduced.strip() or text.strip()


def _factor_set(text: str) -> set[str]:
    return {name for name, pattern in _FACTORS.items() if pattern.search(text)}


def _amount(text: str) -> int | None:
    match = _AMOUNT.search(text)
    if not match:
        return None
    return parse_rmb("".join(match.groups())).amount


def _year(text: str) -> int | None:
    match = _YEAR.search(text)
    return int(match.group(1)) if match else None


def structured_similarity(query: str, candidate: str) -> float:
    query_factors, candidate_factors = _factor_set(query), _factor_set(candidate)
    union = query_factors | candidate_factors
    factor_score = len(query_factors & candidate_factors) / len(union) if union else 0.5
    query_amount, candidate_amount = _amount(query), _amount(candidate)
    amount_score = (
        0.5
        if query_amount is None or candidate_amount is None
        else math.exp(-abs(math.log1p(query_amount) - math.log1p(candidate_amount)))
    )
    query_year, candidate_year = _year(query), _year(candidate)
    year_score = (
        0.5
        if query_year is None or candidate_year is None
        else math.exp(-abs(query_year - candidate_year) / 5.0)
    )
    return 0.45 * factor_score + 0.45 * amount_score + 0.10 * year_score


class ChargePartitionedRetriever:
    """Charge-gated BM25 retrieval with deterministic structured reranking."""

    def __init__(
        self,
        partitions: dict[str, LexicalBM25Index] | None = None,
        *,
        min_score: float = 0.45,
        candidate_multiplier: int = 20,
    ) -> None:
        self.partitions = partitions or {}
        self.min_score = min_score
        self.candidate_multiplier = max(1, candidate_multiplier)

    def build(self, chunks: list[CaseChunk]) -> dict[str, int]:
        unique: dict[str, CaseChunk] = {}
        for chunk in chunks:
            fingerprint = _normalized_text(chunk.text)
            if not fingerprint:
                continue
            current = unique.get(fingerprint)
            unique[fingerprint] = (
                chunk
                if current is None
                else current.model_copy(
                    update={
                        "accusations": sorted(set(current.accusations) | set(chunk.accusations))
                    }
                )
            )
        grouped: dict[str, list[CaseChunk]] = defaultdict(list)
        for chunk in unique.values():
            for accusation in {
                canonical_accusation(value) for value in chunk.accusations if value.strip()
            }:
                grouped[accusation].append(chunk)
        self.partitions = {}
        for accusation, rows in grouped.items():
            index = LexicalBM25Index()
            index.build(rows)
            self.partitions[accusation] = index
        return {name: len(index.chunks) for name, index in self.partitions.items()}

    @staticmethod
    def _allocate(scores: list[LabelScore], top_k: int) -> dict[str, int]:
        if not scores or top_k <= 0:
            return {}
        total = sum(max(item.probability, 0.0) for item in scores)
        if total <= 0:
            return {scores[0].label: top_k}
        raw = {item.label: top_k * item.probability / total for item in scores}
        allocation = {label: int(value) for label, value in raw.items()}
        for item in scores:
            if item.probability > 0 and allocation[item.label] == 0:
                allocation[item.label] = 1
        while sum(allocation.values()) > top_k:
            removable = [label for label, count in allocation.items() if count > 1]
            if removable:
                allocation[min(removable, key=lambda label: raw[label] - allocation[label])] -= 1
            else:
                active = [item for item in scores if allocation[item.label] > 0]
                allocation[min(active, key=lambda item: item.probability).label] = 0
        while sum(allocation.values()) < top_k:
            best = max(raw, key=lambda label: raw[label] - allocation[label])
            allocation[best] += 1
        return {label: count for label, count in allocation.items() if count > 0}

    def search_by_accusations(
        self, query: str, accusations: list[LabelScore], top_k: int = 3
    ) -> list[SearchHit]:
        allocation = {
            canonical_accusation(label): quota
            for label, quota in self._allocate(accusations, min(top_k, 3)).items()
        }
        probabilities = {canonical_accusation(item.label): item.probability for item in accusations}
        if not allocation or not probabilities:
            return []
        max_probability = max(probabilities.values()) or 1.0
        candidates: dict[str, tuple[SearchHit, str]] = {}
        for accusation, quota in allocation.items():
            index = self.partitions.get(accusation)
            if index is None:
                continue
            hits = index.search(
                _query_text(query), top_k=max(quota * self.candidate_multiplier, top_k)
            )
            max_bm25 = max((hit.score for hit in hits), default=1.0) or 1.0
            for hit in hits:
                bm25 = max(0.0, hit.score / max_bm25)
                charge = probabilities[accusation] / max_probability
                structured = structured_similarity(query, hit.text)
                cardinality = float((len(probabilities) > 1) == (len(set(hit.accusations)) > 1))
                score = 0.55 * bm25 + 0.25 * charge + 0.15 * structured + 0.05 * cardinality
                reranked = hit.model_copy(
                    update={
                        "score": min(1.0, score),
                        "source_scores": {
                            "bm25_normalized": bm25,
                            "charge_probability": probabilities[accusation],
                            "structured_similarity": structured,
                            "charge_cardinality_match": cardinality,
                        },
                    }
                )
                current = candidates.get(hit.case_id)
                if current is None or reranked.score > current[0].score:
                    candidates[hit.case_id] = (reranked, accusation)
        selected: list[SearchHit] = []
        used: dict[str, int] = defaultdict(int)
        for hit, accusation in sorted(
            candidates.values(), key=lambda item: item[0].score, reverse=True
        ):
            if hit.score < self.min_score or used[accusation] >= allocation[accusation]:
                continue
            selected.append(hit)
            used[accusation] += 1
            if len(selected) == min(top_k, 3):
                break
        return selected

    def save(self, output_dir: str | Path) -> None:
        target = Path(output_dir)
        if target.resolve() in {Path("/").resolve(), Path.cwd().resolve()}:
            raise ValueError("refusing to replace a broad index path")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        mapping = {}
        for accusation, index in self.partitions.items():
            directory = _partition_name(accusation)
            index.save(target / directory)
            mapping[accusation] = directory
        (target / "manifest.json").write_text(
            json.dumps(
                {
                    "index_type": "charge_partitioned_bm25_structured_reranker",
                    "partitions": mapping,
                    "min_score": self.min_score,
                    "candidate_multiplier": self.candidate_multiplier,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: str | Path) -> ChargePartitionedRetriever:
        source = Path(output_dir)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        partitions = {
            accusation: LexicalBM25Index.load(source / directory)
            for accusation, directory in manifest["partitions"].items()
        }
        return cls(
            partitions,
            min_score=float(manifest.get("min_score", 0.45)),
            candidate_multiplier=int(manifest.get("candidate_multiplier", 20)),
        )
