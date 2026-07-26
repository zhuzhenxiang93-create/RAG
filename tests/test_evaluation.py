from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.evaluation.metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.runner import EvaluationRunner


class EvaluationTest(unittest.TestCase):
    def test_retrieval_metrics(self) -> None:
        retrieved = ["wrong.md", "gold.md"]
        self.assertEqual(recall_at_k(retrieved, ["gold.md"], 1), 0.0)
        self.assertEqual(recall_at_k(retrieved, ["gold.md"], 2), 1.0)
        self.assertEqual(reciprocal_rank(retrieved, ["gold.md"]), 0.5)
        self.assertGreater(ndcg_at_k(retrieved, ["gold.md"], 2), 0.0)

    def test_benchmark_writes_raw_predictions_and_warning(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            result = EvaluationRunner(project_root, Path(directory)).run("lite_v1")
            artifact = Path(result.artifact_path)
            payload = __import__("json").loads(artifact.read_text(encoding="utf-8"))

        self.assertEqual(result.question_count, 14)
        self.assertEqual(result.answerable_count, 12)
        self.assertIn("adaptive", result.retrieval_metrics)
        self.assertIn("decision_accuracy", result.qa_metrics)
        self.assertIn("raw_predictions", payload)
        self.assertIn("Do not present", payload["warning"])


if __name__ == "__main__":
    unittest.main()
