from pathlib import Path
import unittest

from app.core.config import Settings


class SettingsTest(unittest.TestCase):
    def test_lite_defaults(self) -> None:
        settings = Settings(project_root=Path.cwd())
        self.assertEqual(settings.app_mode, "lite")
        self.assertEqual(settings.llm_api_key, "")
        self.assertFalse(settings.ocr_enabled)
        self.assertTrue(settings.intent_enabled)
        self.assertEqual(settings.intent_backend, "lite")

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Settings(project_root=Path.cwd(), app_mode="invalid")

    def test_invalid_intent_backend_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Settings(project_root=Path.cwd(), intent_backend="remote")


if __name__ == "__main__":
    unittest.main()
