from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


class HealthTest(unittest.TestCase):
    def test_health_does_not_require_models_or_secrets(self) -> None:
        app = create_app(Settings(project_root=Path.cwd(), app_mode="lite"))
        with TestClient(app) as client:
            response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["mode"], "lite")
        self.assertFalse(body["capabilities"]["llm"])


if __name__ == "__main__":
    unittest.main()
