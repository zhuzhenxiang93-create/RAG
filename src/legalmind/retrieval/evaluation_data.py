from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

from legalmind.schemas import SearchHit

_FEATURE_PATTERNS = {
    "amount": re.compile(r"(?:人民币|价值|赃款|金额|获利|损失).{0,12}\d|\d+(?:万|元)"),
    "injury_result": re.compile(r"轻伤|重伤|死亡|伤残|抢救无效|法医"),
    "joint_crime": re.compile(r"共同|伙同|同案|召集|纠集|多人|主犯|从犯"),
    "attempted": re.compile(r"未遂|中止|预备|未得逞"),
    "intentional": re.compile(r"故意|明知|非法占有|预谋|蓄意"),
    "negligent": re.compile(r"过失|疏忽|应当预见|未尽"),
}


class SearchIndex(Protocol):
    def search(self, query: str, top_k: int = 20) -> list[SearchHit]: ...


def semantic_dimensions(text: str) -> list[str]:
    return [name for name, pattern in _FEATURE_PATTERNS.items() if pattern.search(text)]


def frequency_band(accusations: list[str], support: dict[str, int]) -> str:
    if not accusations:
        return "unknown"
    minimum = min(support.get(label, 0) for label in accusations)
    if minimum <= 72:
        return "extreme_tail"
    if minimum <= 1891:
        return "tail"
    return "head"


def query_tags(row: dict, support: dict[str, int]) -> list[str]:
    tags = {frequency_band(row["labels"]["accusations"], support)}
    if len(row["labels"]["accusations"]) > 1:
        tags.add("multi_label")
    if len(row["fact"]) >= 1120:
        tags.add("long_text")
    tags.update(semantic_dimensions(row["fact"]))
    return sorted(tags)


def select_diverse_queries(
    rows: list[dict],
    support: dict[str, int],
    target_size: int = 200,
    seed: int = 42,
) -> list[dict]:
    if target_size <= 0:
        return []
    rng = random.Random(seed)
    candidates = [(row, query_tags(row, support)) for row in rows]
    by_tag: dict[str, list[int]] = {}
    for index, (_, tags) in enumerate(candidates):
        for tag in tags:
            by_tag.setdefault(tag, []).append(index)
    for indices in by_tag.values():
        rng.shuffle(indices)

    selected: list[int] = []
    used: set[int] = set()
    ordered_tags = [
        "extreme_tail",
        "tail",
        "head",
        "multi_label",
        "long_text",
        "amount",
        "injury_result",
        "joint_crime",
        "attempted",
        "intentional",
        "negligent",
    ]
    while len(selected) < min(target_size, len(candidates)):
        progress = False
        for tag in ordered_tags:
            indices = by_tag.get(tag, [])
            while indices and indices[-1] in used:
                indices.pop()
            if indices:
                index = indices.pop()
                used.add(index)
                selected.append(index)
                progress = True
                if len(selected) >= target_size:
                    break
        if not progress:
            break
    if len(selected) < min(target_size, len(candidates)):
        remaining = [index for index in range(len(candidates)) if index not in used]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_size - len(selected)])

    output = []
    for index in selected:
        row, tags = candidates[index]
        output.append(
            {
                "query_id": row["case_id"],
                "source_split": row["source_split"],
                "fact": row["fact"],
                "accusations": list(row["labels"]["accusations"]),
                "query_tags": tags,
                "review_status": "unreviewed",
            }
        )
    return output


def _deduplicate_case_hits(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    output = []
    for hit in hits:
        if hit.case_id not in seen:
            seen.add(hit.case_id)
            output.append(hit)
    return output


def build_silver_candidates(
    queries: list[dict],
    index: SearchIndex,
    candidates_per_query: int = 100,
    positives_per_query: int = 3,
    negatives_per_query: int = 3,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    candidate_pool: dict[str, dict] = {}
    qrels: list[dict] = []
    review_rows: list[dict] = []
    reason_counts: Counter[str] = Counter()
    queries_without_positive = 0

    for query in queries:
        query_labels = set(query["accusations"])
        query_dimensions = set(semantic_dimensions(query["fact"]))
        hits = [
            hit
            for hit in _deduplicate_case_hits(index.search(query["fact"], candidates_per_query))
            if hit.case_id != query["query_id"]
        ]
        positives = [hit for hit in hits if query_labels.intersection(hit.accusations)]
        negatives = [hit for hit in hits if not query_labels.intersection(hit.accusations)]
        selected = positives[:positives_per_query] + negatives[:negatives_per_query]
        if not positives:
            queries_without_positive += 1

        for rank, hit in enumerate(selected, start=1):
            shared_labels = sorted(query_labels.intersection(hit.accusations))
            candidate_dimensions = set(semantic_dimensions(hit.text))
            shared_dimensions = sorted(query_dimensions.intersection(candidate_dimensions))
            if shared_labels and shared_dimensions:
                relevance = 2
                reason = "shared_charge_and_fact_dimension"
            elif shared_labels:
                relevance = 1
                reason = "shared_charge_only"
            else:
                relevance = 0
                reason = "surface_similar_different_charge"
            reason_counts[reason] += 1
            candidate_pool.setdefault(
                hit.case_id,
                {
                    "candidate_id": hit.case_id,
                    "text": hit.text,
                    "accusations": list(hit.accusations),
                    "relevant_articles": list(hit.relevant_articles),
                    # The evaluation index is deliberately built from train chunks only.
                    "source_split": "train",
                    "review_status": "unreviewed",
                },
            )
            qrel = {
                "query_id": query["query_id"],
                "candidate_id": hit.case_id,
                "silver_relevance": relevance,
                "shared_accusations": shared_labels,
                "shared_dimensions": shared_dimensions,
                "automatic_reason": reason,
                "retrieval_rank": rank,
                "retrieval_score": hit.score,
                "judgment_source": "silver_proxy",
                "review_status": "unreviewed",
            }
            qrels.append(qrel)
            review_rows.append(
                {
                    **qrel,
                    "query_fact": query["fact"],
                    "query_accusations": "|".join(query["accusations"]),
                    "candidate_text": hit.text,
                    "candidate_accusations": "|".join(hit.accusations),
                    "human_relevance": "",
                    "human_reason": "",
                    "reviewer": "",
                    "reviewed_at": "",
                }
            )

    stats = {
        "queries": len(queries),
        "unique_candidates": len(candidate_pool),
        "silver_qrels": len(qrels),
        "queries_without_positive_candidate": queries_without_positive,
        "automatic_reason_distribution": dict(reason_counts),
        "judgment_source": "silver_proxy_unreviewed",
    }
    return list(candidate_pool.values()), qrels, review_rows, stats


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_line(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
