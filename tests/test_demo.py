from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


class DemoWorkflowTest(unittest.TestCase):
    def test_bootstrap_is_lite_only_and_idempotent(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=project_root, data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                first = client.post("/api/demo/bootstrap")
                second = client.post("/api/demo/bootstrap")

            self.assertEqual(first.status_code, 200)
            self.assertEqual(len(first.json()["added"]), 7)
            self.assertEqual(first.json()["index"]["document_count"], 7)
            self.assertEqual(second.json()["added"], [])
            self.assertEqual(len(second.json()["skipped"]), 7)

    def test_bootstrap_is_disabled_in_full_mode(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            settings = Settings(
                project_root=project_root,
                data_dir=Path(directory),
                app_mode="full",
            )
            with TestClient(create_app(settings)) as client:
                response = client.post("/api/demo/bootstrap")

            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.json()["error"]["code"], "demo_disabled")


if __name__ == "__main__":
    unittest.main()
