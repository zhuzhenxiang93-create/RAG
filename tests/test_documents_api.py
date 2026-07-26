from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


class DocumentsApiTest(unittest.TestCase):
    def test_upload_list_detail_and_delete(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(
                project_root=Path.cwd(),
                data_dir=Path(directory),
                app_mode="lite",
                max_upload_mb=1,
            )
            with TestClient(create_app(settings)) as client:
                uploaded = client.post(
                    "/api/documents/upload",
                    files={"file": ("policy.md", "# Policy\nEvidence must be cited.", "text/markdown")},
                )
                self.assertEqual(uploaded.status_code, 201, uploaded.text)
                record = uploaded.json()
                document_id = record["document_id"]
                self.assertEqual(record["status"], "ready")
                self.assertGreater(record["child_chunk_count"], 0)

                listing = client.get("/api/documents")
                self.assertEqual(listing.status_code, 200)
                self.assertEqual(listing.json()["total"], 1)

                detail = client.get("/api/documents/{}".format(document_id))
                self.assertEqual(detail.status_code, 200)
                self.assertTrue(detail.json()["elements"])
                self.assertTrue(detail.json()["chunks"])

                deleted = client.delete("/api/documents/{}".format(document_id))
                self.assertEqual(deleted.status_code, 204)
                self.assertEqual(client.get("/api/documents").json()["total"], 0)

    def test_path_traversal_filename_is_reduced_to_basename(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=Path.cwd(), data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": ("../policy.txt", b"safe content", "text/plain")},
                )
            self.assertEqual(response.status_code, 201)
            self.assertEqual(response.json()["filename"], "policy.txt")

    def test_unsupported_upload_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            settings = Settings(project_root=Path.cwd(), data_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": ("payload.exe", b"MZ", "application/octet-stream")},
                )
            self.assertEqual(response.status_code, 415)
            self.assertEqual(response.json()["error"]["code"], "unsupported_file_type")


if __name__ == "__main__":
    unittest.main()
