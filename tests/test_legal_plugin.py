import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.plugins.legal.assets import load_labels, validate_assets
from app.plugins.legal.training import LegalTrainingConfig


class LegalPluginTest(unittest.TestCase):
    def test_disabled_plugin_does_not_import_model_dependencies(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=Path.cwd(), data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                status = client.get("/api/plugins/legal/status")
                classify = client.post(
                    "/api/plugins/legal/classify",
                    json={"text": "测试案情", "top_k": 3},
                )

        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["state"], "disabled")
        self.assertEqual(classify.status_code, 503)
        self.assertEqual(
            classify.json()["error"]["code"], "legal_plugin_unavailable"
        )

    def test_labels_must_be_contiguous(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps({"A": 0, "B": 2}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_labels(path)

    def test_assets_are_validated_without_loading_models(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            score = root / "score_weights.pt"
            score.write_bytes(b"placeholder")
            labels = root / "labels.json"
            labels.write_text(json.dumps({"A": 0, "B": 1}), encoding="utf-8")
            assets = validate_assets(
                base_model="local-or-hf-model",
                adapter_path=str(adapter),
                score_weights_path=str(score),
                labels_path=str(labels),
            )

        self.assertEqual(len(assets.labels), 2)
        self.assertEqual(assets.labels_by_id[1], "B")

    def test_training_config_is_deterministic_and_validated(self) -> None:
        config = LegalTrainingConfig(
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            output_dir="artifacts/legal",
            num_labels=202,
        )
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.target_modules, ["q_proj", "v_proj"])
        with self.assertRaises(ValueError):
            LegalTrainingConfig(base_model="x", output_dir="y", num_labels=1)


if __name__ == "__main__":
    unittest.main()
