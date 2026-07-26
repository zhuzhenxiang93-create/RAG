from pathlib import Path
import unittest

from app.core.config import Settings


class SettingsTest(unittest.TestCase):
    def test_lite_defaults(self) -> None:
        settings = Settings(project_root=Path.cwd())
        self.assertEqual(settings.app_mode, "lite")
        self.assertEqual(settings.llm_api_key, "")
        self.assertFalse(settings.ocr_enabled)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Settings(project_root=Path.cwd(), app_mode="invalid")


if __name__ == "__main__":
    unittest.main()
