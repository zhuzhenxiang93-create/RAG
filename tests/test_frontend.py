from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


class FrontendTest(unittest.TestCase):
    def test_root_redirects_to_local_workbench(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=project_root, data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                root = client.get("/", follow_redirects=False)
                page = client.get("/ui/")
                script = client.get("/ui/app.js")

            self.assertEqual(root.status_code, 307)
            self.assertEqual(root.headers["location"], "/ui/")
            self.assertIn("文档智能工作台", page.text)
            self.assertIn('const API = "/api"', script.text)
            self.assertNotIn("https://", page.text)


if __name__ == "__main__":
    unittest.main()
