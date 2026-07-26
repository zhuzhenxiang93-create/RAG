"""Explainable query routing for adaptive retrieval."""

import re
from typing import List

from app.retrieval.tokenizer import tokenize
from app.schemas.search import RetrievalPlan


class QueryRouter:
    """Rule-first router that remains deterministic and inspectable."""

    _identifier = re.compile(
        r"\b(?:[A-Z]{2,}[-_/]?\d+|\d{4,}|[A-Za-z]+\d+[A-Za-z0-9_-]*|"
        r"[A-Za-z]+@[A-Za-z0-9]+)\b"
    )
    _comparison = ("比较", "对比", "差异", "不同", "分别", "versus", " vs ", "compare")
    _table = ("表格", "哪一行", "哪一列", "总计", "同比", "环比", "占比", "多少", "数值")
    _summary = ("总结", "概述", "摘要", "主要内容", "归纳", "summarize", "summary")
    _out_of_scope = ("天气", "写首诗", "讲笑话", "星座", "彩票")

    def route(self, query: str) -> RetrievalPlan:
        lowered = query.lower()
        keywords = self._keywords(query)

        if any(marker in lowered for marker in self._out_of_scope):
            return RetrievalPlan(
                query_type="out_of_scope",
                confidence=0.9,
                keywords=keywords,
                bm25_weight=0.0,
                dense_weight=0.0,
                use_reranker=False,
                reason="Query matches configured out-of-scope intent.",
            )
        if any(marker in lowered for marker in self._comparison):
            return RetrievalPlan(
                query_type="comparison",
                confidence=0.84,
                keywords=keywords,
                bm25_weight=0.7,
                dense_weight=1.0,
                use_reranker=True,
                reason="Comparison markers require evidence from multiple passages.",
            )
        if any(marker in lowered for marker in self._summary):
            return RetrievalPlan(
                query_type="summarization",
                confidence=0.82,
                keywords=keywords,
                bm25_weight=0.4,
                dense_weight=1.0,
                use_reranker=False,
                reason="Summary intent favors broad semantic coverage.",
            )
        if any(marker in lowered for marker in self._table):
            return RetrievalPlan(
                query_type="table_qa",
                confidence=0.78,
                keywords=keywords,
                bm25_weight=1.0,
                dense_weight=0.75,
                use_reranker=True,
                reason="Numeric or table markers favor exact tokens plus semantic recall.",
            )
        if self._identifier.search(query) or any(char in query for char in ('"', "“", "”")):
            return RetrievalPlan(
                query_type="exact_lookup",
                confidence=0.88,
                keywords=keywords,
                bm25_weight=1.2,
                dense_weight=0.45,
                use_reranker=True,
                reason="Identifier or quoted phrase detected.",
            )
        return RetrievalPlan(
            query_type="semantic_qa",
            confidence=0.68,
            keywords=keywords,
            bm25_weight=0.65,
            dense_weight=1.0,
            use_reranker=True,
            reason="Default semantic question route.",
        )

    @staticmethod
    def _keywords(query: str) -> List[str]:
        seen = set()
        result = []
        for token in tokenize(query):
            if len(token) < 2 or token in seen:
                continue
            seen.add(token)
            result.append(token)
            if len(result) == 12:
                break
        return result
