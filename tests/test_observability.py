import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.logging import JsonFormatter
from app.main import create_app


class ObservabilityTest(unittest.TestCase):
    def test_request_id_and_server_timing_are_returned(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=project_root, data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                response = client.get(
                    "/api/health", headers={"X-Request-ID": "interview-demo-01"}
                )

            self.assertEqual(response.headers["x-request-id"], "interview-demo-01")
            self.assertTrue(response.headers["server-timing"].startswith("app;dur="))

    def test_invalid_request_id_is_replaced(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=project_root, data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                response = client.get(
                    "/api/health", headers={"X-Request-ID": "unsafe id with spaces"}
                )

            self.assertNotEqual(response.headers["x-request-id"], "unsafe id with spaces")
            self.assertEqual(len(response.headers["x-request-id"]), 32)

    def test_diagnostics_excludes_secrets_and_paths(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            settings = Settings(
                project_root=project_root,
                data_dir=Path(directory),
                llm_api_key="must-not-leak",
            )
            with TestClient(create_app(settings)) as client:
                payload = client.get("/api/diagnostics").json()

            serialized = json.dumps(payload)
            self.assertNotIn("must-not-leak", serialized)
            self.assertNotIn(str(project_root), serialized)
            self.assertEqual(payload["document_count"], 0)
            self.assertFalse(payload["index_ready"])

    def test_json_formatter_emits_structured_fields(self) -> None:
        import logging

        record = logging.LogRecord(
            name="docmind.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="request_completed",
            args=(),
            exc_info=None,
        )
        record.request_id = "req-1"
        record.status_code = 200
        payload = json.loads(JsonFormatter().format(record))
        self.assertEqual(payload["request_id"], "req-1")
        self.assertEqual(payload["status_code"], 200)


if __name__ == "__main__":
    unittest.main()
