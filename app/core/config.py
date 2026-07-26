"""Environment-backed application configuration."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Optional


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings with safe Lite-mode defaults."""

    project_root: Path
    app_mode: str = "lite"
    api_prefix: str = "/api"
    log_level: str = "INFO"
    log_format: str = "json"
    data_dir: Optional[Path] = None
    max_upload_mb: int = 50
    ocr_enabled: bool = False
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    embedding_provider: str = "lite"
    reranker_enabled: bool = False
    legal_enabled: bool = False
    legal_base_model: str = ""
    legal_adapter_path: str = ""
    legal_score_weights_path: str = ""
    legal_labels_path: str = ""
    legal_device: str = "auto"
    legal_max_length: int = 512

    def __post_init__(self) -> None:
        mode = self.app_mode.lower()
        if mode not in {"lite", "full"}:
            raise ValueError("DOCMIND_APP_MODE must be 'lite' or 'full'")
        object.__setattr__(self, "app_mode", mode)
        log_format = self.log_format.lower()
        if log_format not in {"json", "text"}:
            raise ValueError("DOCMIND_LOG_FORMAT must be 'json' or 'text'")
        object.__setattr__(self, "log_format", log_format)
        if self.data_dir is None:
            object.__setattr__(self, "data_dir", self.project_root / "data")

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(__file__).resolve().parents[2]
        data_value = os.getenv("DOCMIND_DATA_DIR", "").strip()
        return cls(
            project_root=root,
            app_mode=os.getenv("DOCMIND_APP_MODE", "lite"),
            api_prefix=os.getenv("DOCMIND_API_PREFIX", "/api"),
            log_level=os.getenv("DOCMIND_LOG_LEVEL", "INFO"),
            log_format=os.getenv("DOCMIND_LOG_FORMAT", "json"),
            data_dir=Path(data_value).expanduser() if data_value else None,
            max_upload_mb=int(os.getenv("DOCMIND_MAX_UPLOAD_MB", "50")),
            ocr_enabled=_as_bool(os.getenv("DOCMIND_OCR_ENABLED")),
            llm_base_url=os.getenv("DOCMIND_LLM_BASE_URL", ""),
            llm_api_key=os.getenv("DOCMIND_LLM_API_KEY", ""),
            llm_model=os.getenv("DOCMIND_LLM_MODEL", ""),
            embedding_provider=os.getenv("DOCMIND_EMBEDDING_PROVIDER", "lite"),
            reranker_enabled=_as_bool(os.getenv("DOCMIND_RERANKER_ENABLED")),
            legal_enabled=_as_bool(os.getenv("DOCMIND_LEGAL_ENABLED")),
            legal_base_model=os.getenv("DOCMIND_LEGAL_BASE_MODEL", ""),
            legal_adapter_path=os.getenv("DOCMIND_LEGAL_ADAPTER_PATH", ""),
            legal_score_weights_path=os.getenv(
                "DOCMIND_LEGAL_SCORE_WEIGHTS_PATH", ""
            ),
            legal_labels_path=os.getenv("DOCMIND_LEGAL_LABELS_PATH", ""),
            legal_device=os.getenv("DOCMIND_LEGAL_DEVICE", "auto"),
            legal_max_length=int(os.getenv("DOCMIND_LEGAL_MAX_LENGTH", "512")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached immutable settings."""
    return Settings.from_env()
