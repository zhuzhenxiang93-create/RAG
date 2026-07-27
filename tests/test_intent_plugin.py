import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.evaluation.classification import classification_metrics
from app.main import create_app
from app.plugins.intent.assets import load_taxonomy, validate_assets
from app.plugins.intent.training import IntentTrainingConfig


class IntentPluginTest(unittest.TestCase):
    def test_lite_classifier_routes_common_queries(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=Path.cwd(), data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                status = client.get("/api/plugins/intent/status")
                response = client.post(
                    "/api/plugins/intent/classify",
                    json={"text": "我的订单为什么还没有配送？", "top_k": 3},
                )

        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["state"], "lite_ready")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["predictions"][0]["intent"], "order_status")
        self.assertEqual(body["predictions"][0]["domain"], "commerce")
        self.assertEqual(body["route"]["knowledge_base"], "commerce_support")
        self.assertEqual(body["backend"], "lite")

    def test_intent_route_is_visible_in_adaptive_search(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=Path.cwd(), data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                client.post(
                    "/api/documents/upload",
                    files={
                        "file": (
                            "refund.md",
                            "# 退款流程\n订单支付后七天内可以提交退款申请。",
                            "text/markdown",
                        )
                    },
                )
                response = client.post(
                    "/api/search",
                    json={"query": "订单如何申请退款？", "strategy": "adaptive"},
                )

        self.assertEqual(response.status_code, 200, response.text)
        plan = response.json()["plan"]
        self.assertEqual(plan["intent_domain"], "commerce")
        self.assertEqual(plan["knowledge_base"], "commerce_support")
        self.assertEqual(plan["routing_backend"], "lite")
        self.assertIn("intent_router", response.json()["timings_ms"])

    def test_uncertain_lite_route_falls_back_to_all_documents(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=Path.cwd(), data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                response = client.post(
                    "/api/plugins/intent/classify",
                    json={"text": "解释一下这个概念"},
                )
        route = response.json()["route"]
        self.assertTrue(route["abstain"])
        self.assertEqual(route["knowledge_base"], "general_documents")

    def test_lora_assets_and_training_config_are_validated(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            taxonomy = root / "taxonomy.json"
            taxonomy.write_text(
                json.dumps(
                    {
                        "intents": {"refund": 0, "order_status": 1},
                        "intent_to_domain": {
                            "refund": "commerce",
                            "order_status": "commerce",
                        },
                    }
                ),
                encoding="utf-8",
            )
            assets = validate_assets(
                base_model="Qwen/Qwen2.5-1.5B",
                adapter_path=str(adapter),
                labels_path=str(taxonomy),
            )
            intents, domains = load_taxonomy(taxonomy)
        self.assertEqual(assets.intents_by_id[1], "order_status")
        self.assertEqual(domains["refund"], "commerce")
        self.assertEqual(intents["refund"], 0)
        config = IntentTrainingConfig(
            base_model="Qwen/Qwen2.5-1.5B",
            output_dir="artifacts/intent",
            num_labels=60,
        )
        self.assertEqual(config.rank, 16)
        self.assertEqual(config.target_modules, ["q_proj", "v_proj"])

    def test_classification_metrics_do_not_require_sklearn(self) -> None:
        metrics = classification_metrics(
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0.9, 0.6, 0.8, 0.7],
            label_count=2,
        )
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], 0.733333)
        self.assertIn("ece", metrics)
        self.assertEqual(len(metrics["confusion_matrix"]), 2)


if __name__ == "__main__":
    unittest.main()
