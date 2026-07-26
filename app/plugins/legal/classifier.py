"""Lazy PEFT sequence classifier for the legal plugin."""

from importlib.util import find_spec
import inspect
from pathlib import Path
import threading
from typing import List

from app.core.config import Settings
from app.plugins.legal.assets import LegalAssets, safe_label_count, validate_assets
from app.schemas.legal import (
    LegalClassificationResponse,
    LegalPluginStatus,
    LegalPrediction,
)


class LegalClassifier:
    """Load the base model, LoRA adapter and classification head on first use."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._model = None
        self._tokenizer = None
        self._assets = None
        self._device = None

    def status(self) -> LegalPluginStatus:
        dependencies = {
            name: find_spec(name) is not None
            for name in ("torch", "transformers", "peft")
        }
        configured = {
            "base_model": bool(self.settings.legal_base_model),
            "adapter": bool(self.settings.legal_adapter_path),
            "score_weights": bool(self.settings.legal_score_weights_path),
            "labels": bool(self.settings.legal_labels_path),
        }
        label_count = safe_label_count(self.settings.legal_labels_path)
        if not self.settings.legal_enabled:
            state = "disabled"
            message = "Legal plugin is disabled."
        elif not all(configured.values()):
            state = "not_configured"
            message = "Legal plugin configuration is incomplete."
        elif not all(dependencies.values()):
            state = "dependencies_missing"
            message = "Install torch, transformers and peft to enable inference."
        elif self._model is not None:
            state = "loaded"
            message = "Legal classifier is loaded."
        else:
            try:
                validate_assets(
                    base_model=self.settings.legal_base_model,
                    adapter_path=self.settings.legal_adapter_path,
                    score_weights_path=self.settings.legal_score_weights_path,
                    labels_path=self.settings.legal_labels_path,
                )
                state = "ready"
                message = "Legal assets are valid and will be loaded on first inference."
            except ValueError as exc:
                state = "not_configured"
                message = str(exc)
        return LegalPluginStatus(
            enabled=self.settings.legal_enabled,
            state=state,
            dependencies=dependencies,
            configured_assets=configured,
            label_count=label_count,
            device=self._device,
            message=message,
        )

    def predict(self, text: str, top_k: int) -> LegalClassificationResponse:
        self._ensure_loaded()
        import torch

        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=self.settings.legal_max_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(self._device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            logits = self._model(**encoded).logits[0]
            probabilities = torch.softmax(logits.float(), dim=-1)
            count = min(top_k, len(self._assets.labels))
            values, indices = torch.topk(probabilities, k=count)

        labels_by_id = self._assets.labels_by_id
        predictions: List[LegalPrediction] = []
        for value, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
            identifier = int(index)
            predictions.append(
                LegalPrediction(
                    label=labels_by_id[identifier],
                    label_id=identifier,
                    probability=round(float(value), 6),
                )
            )
        return LegalClassificationResponse(
            predictions=predictions,
            model=self._assets.base_model,
            adapter=self._assets.adapter_path.name,
            device=str(self._device),
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            status = self.status()
            if status.state not in {"ready", "loaded"}:
                raise RuntimeError(status.message)

            import torch
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            assets = validate_assets(
                base_model=self.settings.legal_base_model,
                adapter_path=self.settings.legal_adapter_path,
                score_weights_path=self.settings.legal_score_weights_path,
                labels_path=self.settings.legal_labels_path,
            )
            device = self._select_device(torch)
            tokenizer = AutoTokenizer.from_pretrained(
                assets.base_model, trust_remote_code=True
            )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                assets.base_model,
                trust_remote_code=True,
                num_labels=len(assets.labels),
            )
            model = PeftModel.from_pretrained(base_model, str(assets.adapter_path))
            head = getattr(model, "score", None)
            if head is None and hasattr(model, "base_model"):
                head = getattr(model.base_model, "score", None)
            if head is None:
                raise RuntimeError("Sequence-classification head 'score' was not found")

            load_parameters = inspect.signature(torch.load).parameters
            load_kwargs = {"map_location": "cpu"}
            if "weights_only" in load_parameters:
                load_kwargs["weights_only"] = True
            state = torch.load(str(assets.score_weights_path), **load_kwargs)
            head.load_state_dict(state, strict=True)
            model = model.to(device).eval()

            self._assets = assets
            self._tokenizer = tokenizer
            self._model = model
            self._device = device

    def _select_device(self, torch):
        requested = self.settings.legal_device.lower()
        if requested == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(requested)
