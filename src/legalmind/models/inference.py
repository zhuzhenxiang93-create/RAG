from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from legalmind.models.loading import configure_padding
from legalmind.models.metrics import sigmoid
from legalmind.schemas import ClassificationResult, LabelScore
from legalmind.training.long_text import truncate_fact


def validate_sequence_classifier_adapter(adapter_path: str | Path) -> dict:
    adapter_config = Path(adapter_path) / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"missing PEFT adapter config: {adapter_config}")
    metadata = json.loads(adapter_config.read_text(encoding="utf-8"))
    saved_heads = set(metadata.get("modules_to_save") or [])
    if not saved_heads.intersection({"score", "classifier"}):
        raise ValueError(
            "PEFT sequence-classification adapter does not contain a saved classifier head"
        )
    return metadata


class ChargeClassifier:
    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        id_to_label: dict[int, str],
        thresholds_path: str | None = None,
        max_length: int = 1024,
        truncation_strategy: str = "head_tail",
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        validate_sequence_classifier_adapter(adapter_path)

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=len(id_to_label),
            problem_type="multi_label_classification",
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        configure_padding(self.tokenizer, base)
        self.model = PeftModel.from_pretrained(base, adapter_path).eval()
        self.id_to_label = id_to_label
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.thresholds = np.full(len(id_to_label), 0.5, dtype=np.float32)
        threshold_file = Path(thresholds_path) if thresholds_path else None
        if threshold_file and threshold_file.exists():
            values = json.loads(threshold_file.read_text(encoding="utf-8"))
            for key, value in values.items():
                self.thresholds[int(key)] = float(value)

    def predict(self, text: str, top_k: int = 5) -> ClassificationResult:
        model_input = truncate_fact(
            self.tokenizer, text, self.max_length, strategy=self.truncation_strategy
        )
        inputs = self.tokenizer(
            model_input,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits.float().cpu().numpy()[0]
        probabilities = sigmoid(logits)
        selected = np.where(probabilities >= self.thresholds)[0].tolist()
        used_fallback = not selected
        if not selected:
            selected = np.argsort(-probabilities)[: max(1, top_k)].tolist()
        selected = sorted(selected, key=lambda index: probabilities[index], reverse=True)[:top_k]
        scores = [
            LabelScore(
                label_id=index,
                label=self.id_to_label[index],
                probability=float(probabilities[index]),
            )
            for index in selected
        ]
        return ClassificationResult(
            labels=scores,
            thresholds={index: float(self.thresholds[index]) for index in selected},
            used_fallback=used_fallback,
            max_probability=float(probabilities.max()),
        )
