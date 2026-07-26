from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


class TrustworthyQATest(unittest.TestCase):
    def test_supported_answer_contains_citation_and_breakdown(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=Path.cwd(), data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                client.post(
                    "/api/documents/upload",
                    files={
                        "file": (
                            "security.md",
                            "# Security\nPolicy ABC-2025 requires secrets to be stored "
                            "in environment variables. Plaintext keys are prohibited.",
                            "text/markdown",
                        )
                    },
                )
                response = client.post(
                    "/api/chat",
                    json={"query": "Policy ABC-2025 如何保存 secrets？", "top_k": 5},
                )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["decision"], "answer")
        self.assertGreater(body["confidence"], 0.5)
        self.assertTrue(body["citations"])
        self.assertIn("[E1]", body["answer"])
        self.assertIn("evidence_relevance", body["confidence_breakdown"])

    def test_missing_evidence_abstains(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=Path.cwd(), data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                client.post(
                    "/api/documents/upload",
                    files={"file": ("policy.md", "# Security\nUse environment variables.", "text/markdown")},
                )
                response = client.post(
                    "/api/chat",
                    json={"query": "量子发动机的推力是多少？", "top_k": 5},
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["decision"], "abstain")
        self.assertEqual(body["confidence"], 0.0)
        self.assertEqual(body["citations"], [])

    def test_numeric_conflict_requires_clarification(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=Path.cwd(), data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                client.post(
                    "/api/documents/upload",
                    files={
                        "file": (
                            "policy-v1.md",
                            "# Retention\n用户数据保留期限为 30 天。",
                            "text/markdown",
                        )
                    },
                )
                client.post(
                    "/api/documents/upload",
                    files={
                        "file": (
                            "policy-v2.md",
                            "# Retention\n用户数据保留期限为 90 天。",
                            "text/markdown",
                        )
                    },
                )
                response = client.post(
                    "/api/chat",
                    json={"query": "用户数据保留期限是多少天？", "top_k": 5},
                )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["decision"], "clarify")
        self.assertTrue(body["conflicts"])
        self.assertEqual(body["conflicts"][0]["conflict_type"], "numeric")
        self.assertGreater(body["confidence_breakdown"]["conflict_penalty"], 0)

    def test_shared_date_does_not_hide_different_day_measurements(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=Path.cwd(), data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                for filename, text in (
                    ("v1.md", "# 数据保留\n2024 年 1 月 1 日生效，日志保留 30 天。"),
                    (
                        "v2.md",
                        "# 数据保留\n2025 年 1 月 1 日生效，替代 2024 版本，日志保留 90 天。",
                    ),
                ):
                    client.post(
                        "/api/documents/upload",
                        files={"file": (filename, text, "text/markdown")},
                    )
                response = client.post(
                    "/api/chat",
                    json={"query": "比较日志保留期限的差异", "top_k": 5},
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["decision"], "clarify")
        self.assertTrue(response.json()["conflicts"])


if __name__ == "__main__":
    unittest.main()
