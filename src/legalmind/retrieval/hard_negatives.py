from __future__ import annotations

from collections import Counter

from legalmind.retrieval.evaluation_data import semantic_dimensions
from legalmind.retrieval.lexical import LexicalBM25Index


def classify_negative_type(
    query_text: str,
    candidate_text: str,
    query_labels: set[str],
    candidate_labels: set[str],
) -> str:
    query_features = set(semantic_dimensions(query_text))
    candidate_features = set(semantic_dimensions(candidate_text))
    shared_labels = query_labels.intersection(candidate_labels)
    if shared_labels and not query_features.intersection(candidate_features):
        return "same_charge_different_behavior"
    if ("intentional" in query_features) != ("intentional" in candidate_features):
        return "different_intent"
    if ("negligent" in query_features) != ("negligent" in candidate_features):
        return "intentional_vs_negligent"
    if ("attempted" in query_features) != ("attempted" in candidate_features):
        return "completed_vs_attempted"
    if ("joint_crime" in query_features) != ("joint_crime" in candidate_features):
        return "joint_vs_individual"
    if ("injury_result" in query_features) != ("injury_result" in candidate_features):
        return "different_result"
    if "amount" in query_features and "amount" in candidate_features:
        return "different_amount_band"
    return "similar_fact_different_charge"


def mine_hard_negatives(
    queries: list[dict],
    index: LexicalBM25Index,
    candidates_per_query: int = 50,
    negatives_per_query: int = 3,
) -> tuple[list[dict], dict]:
    rows = []
    types: Counter[str] = Counter()
    for query in queries:
        gold = set(query["labels"]["accusations"])
        hits = [
            hit
            for hit in index.search(query["fact"], candidates_per_query)
            if hit.case_id != query["case_id"]
        ]
        positives = [hit for hit in hits if gold.intersection(hit.accusations)]
        if not positives:
            continue
        positive = positives[0]
        selected = 0
        for hit in hits:
            candidate = set(hit.accusations)
            if hit.case_id == positive.case_id:
                continue
            negative_type = classify_negative_type(query["fact"], hit.text, gold, candidate)
            if gold.intersection(candidate) and negative_type != "same_charge_different_behavior":
                continue
            rows.append(
                {
                    "query_id": query["case_id"],
                    "positive_id": positive.case_id,
                    "positive_labels": sorted(gold),
                    "negative_id": hit.case_id,
                    "negative_labels": sorted(candidate),
                    "negative_type": negative_type,
                    "selection_reason": (
                        "BM25 high-rank candidate selected by charge and semantic-feature contrast"
                    ),
                    "bm25_score": hit.score,
                    "source_split": "train",
                    "review_status": "unreviewed",
                }
            )
            types[negative_type] += 1
            selected += 1
            if selected >= negatives_per_query:
                break
    return rows, {"rows": len(rows), "negative_types": dict(types)}
