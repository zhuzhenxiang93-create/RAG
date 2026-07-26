from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import IndexedChunk
from app.routing.query_router import QueryRouter


class RetrievalTest(unittest.TestCase):
    def test_rrf_does_not_reward_missing_channels(self) -> None:
        first = IndexedChunk("d1", "a.md", "c1", "alpha", "text")
        second = IndexedChunk("d2", "b.md", "c2", "beta", "text")
        fused = reciprocal_rank_fusion([(first, 3.0)], [(second, 0.9)])
        self.assertEqual(set(fused[0].channels), {"bm25"})
        self.assertEqual(set(fused[1].channels), {"dense"})
        self.assertIsNone(fused[0].dense_rank)
        self.assertIsNone(fused[1].bm25_rank)

    def test_router_selects_exact_and_comparison_routes(self) -> None:
        router = QueryRouter()
        exact = router.route("编号 ABC-2025 的要求是什么？")
        metric = router.route("Recall@K 的定义是什么？")
        comparison = router.route("比较两个版本的主要差异")
        self.assertEqual(exact.query_type, "exact_lookup")
        self.assertEqual(metric.query_type, "exact_lookup")
        self.assertGreater(exact.bm25_weight, exact.dense_weight)
        self.assertEqual(comparison.query_type, "comparison")
        self.assertTrue(comparison.use_reranker)

    def test_end_to_end_adaptive_search_exposes_channel_ranks(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=Path.cwd(), data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                client.post(
                    "/api/documents/upload",
                    files={
                        "file": (
                            "security.md",
                            "# Security Policy\nPolicy ID ABC-2025 requires all secrets "
                            "to use environment variables.",
                            "text/markdown",
                        )
                    },
                )
                client.post(
                    "/api/documents/upload",
                    files={
                        "file": (
                            "evaluation.md",
                            "# Evaluation\nSemantic retrieval is evaluated with Recall@5 and MRR.",
                            "text/markdown",
                        )
                    },
                )
                built = client.post("/api/index/build")
                response = client.post(
                    "/api/search",
                    json={
                        "query": "Policy ID ABC-2025 要求如何保存密钥？",
                        "strategy": "adaptive",
                        "top_k": 3,
                    },
                )
                strategy_responses = {
                    strategy: client.post(
                        "/api/search",
                        json={
                            "query": "Policy ID ABC-2025 environment variables",
                            "strategy": strategy,
                            "top_k": 2,
                        },
                    )
                    for strategy in ("bm25", "dense", "rrf", "rrf_rerank", "adaptive")
                }

        self.assertEqual(built.status_code, 200, built.text)
        self.assertEqual(built.json()["document_count"], 2)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["plan"]["query_type"], "exact_lookup")
        self.assertEqual(body["results"][0]["filename"], "security.md")
        self.assertIn("bm25", body["results"][0]["channels"])
        self.assertTrue(body["results"][0]["parent_content"])
        self.assertIn("bm25", body["timings_ms"])
        self.assertIn("dense", body["timings_ms"])
        for strategy, strategy_response in strategy_responses.items():
            self.assertEqual(strategy_response.status_code, 200, strategy)
            strategy_body = strategy_response.json()
            self.assertEqual(strategy_body["strategy"], strategy)
            self.assertTrue(strategy_body["results"], strategy)
            self.assertEqual(strategy_body["results"][0]["filename"], "security.md", strategy)

    def test_out_of_scope_query_returns_no_results(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=Path.cwd(), data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                response = client.post(
                    "/api/search",
                    json={"query": "今天天气怎么样？", "strategy": "adaptive"},
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["plan"]["query_type"], "out_of_scope")
        self.assertEqual(response.json()["results"], [])


if __name__ == "__main__":
    unittest.main()
