from __future__ import annotations

from legalmind.retrieval.evaluation_data import (
    build_silver_candidates,
    select_diverse_queries,
    semantic_dimensions,
)
from legalmind.retrieval.lexical import LexicalBM25Index
from legalmind.schemas import CaseChunk


def row(case_id: str, split: str, fact: str, labels: list[str]) -> dict:
    return {
        "case_id": case_id,
        "source_split": split,
        "fact": fact,
        "labels": {"accusations": labels},
    }


def test_query_selection_is_deterministic_and_validation_only() -> None:
    rows = [
        row(f"q{index}", "validation", f"共同犯罪造成轻伤{index}", ["故意伤害"])
        for index in range(20)
    ]
    first = select_diverse_queries(rows, {"故意伤害": 2000}, 10, seed=7)
    second = select_diverse_queries(rows, {"故意伤害": 2000}, 10, seed=7)
    assert first == second
    assert all(value["source_split"] == "validation" for value in first)
    assert all(value["review_status"] == "unreviewed" for value in first)


def test_semantic_dimensions_are_explainable() -> None:
    dimensions = semantic_dimensions("二人共同故意持刀造成轻伤，损失人民币5000元")
    assert {"amount", "injury_result", "joint_crime", "intentional"}.issubset(dimensions)


def test_silver_qrels_never_claim_manual_review() -> None:
    chunks = [
        CaseChunk(
            chunk_id="p:c0",
            case_id="p",
            text="共同故意持刀造成轻伤",
            chunk_index=0,
            start_char=0,
            end_char=10,
            accusations=["故意伤害"],
            source_split="train",
        ),
        CaseChunk(
            chunk_id="n:c0",
            case_id="n",
            text="持刀秘密窃取财物",
            chunk_index=0,
            start_char=0,
            end_char=8,
            accusations=["盗窃"],
            source_split="train",
        ),
    ]
    index = LexicalBM25Index()
    index.build(chunks)
    queries = [
        {
            "query_id": "q",
            "fact": "共同故意持刀造成轻伤",
            "accusations": ["故意伤害"],
        }
    ]
    candidates, qrels, review_rows, stats = build_silver_candidates(
        queries, index, candidates_per_query=2
    )
    assert candidates
    assert qrels
    assert review_rows
    assert stats["judgment_source"] == "silver_proxy_unreviewed"
    assert all(value["review_status"] == "unreviewed" for value in qrels)
    assert all(value["judgment_source"] == "silver_proxy" for value in qrels)
