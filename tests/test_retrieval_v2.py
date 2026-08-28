from __future__ import annotations

from legalmind.retrieval.knowledge import chinese_number, parse_statute_text
from legalmind.retrieval.lexical import LexicalBM25Index
from legalmind.schemas import CaseChunk


def test_chinese_article_number() -> None:
    assert chinese_number("二百三十四") == 234


def test_statute_parser_retains_provenance() -> None:
    rows = parse_statute_text(
        "第二百三十四条 故意伤害他人身体的，依法处罚。第二百三十五条 过失致人重伤的，依法处罚。",
        "https://example.invalid/official",
        "test",
        "unverified_source",
    )
    assert [row.article_number for row in rows] == [234, 235]
    assert all(row.source_status == "unverified_source" for row in rows)


def test_bm25_returns_relevant_surface_match() -> None:
    chunks = [
        CaseChunk(
            chunk_id="1", case_id="a", text="持刀造成轻伤", chunk_index=0, start_char=0, end_char=7
        ),
        CaseChunk(
            chunk_id="2", case_id="b", text="秘密窃取手机", chunk_index=0, start_char=0, end_char=6
        ),
        CaseChunk(
            chunk_id="3", case_id="c", text="酒后驾驶汽车", chunk_index=0, start_char=0, end_char=6
        ),
    ]
    index = LexicalBM25Index()
    index.build(chunks)
    assert index.search("窃取手机", 1)[0].case_id == "b"
