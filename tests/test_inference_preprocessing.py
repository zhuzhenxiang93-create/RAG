from __future__ import annotations

import numpy as np
import pytest
import torch

from legalmind.models.inference import ChargeClassifier, validate_sequence_classifier_adapter


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.last_text = ""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(token_id) for token_id in token_ids)

    def __call__(self, text: str, **_kwargs) -> FakeBatch:
        self.last_text = text
        return FakeBatch(input_ids=torch.tensor([[1, 2, 3, 4]]))


class FakeModel:
    device = "cpu"

    def __call__(self, **_inputs):
        return type("Output", (), {"logits": torch.tensor([[2.0, -2.0]])})()


def test_online_classifier_uses_training_head_tail_preprocessing() -> None:
    classifier = ChargeClassifier.__new__(ChargeClassifier)
    classifier.torch = torch
    classifier.tokenizer = FakeTokenizer()
    classifier.model = FakeModel()
    classifier.id_to_label = {0: "人工标签甲", 1: "人工标签乙"}
    classifier.max_length = 4
    classifier.truncation_strategy = "head_tail"
    classifier.thresholds = np.asarray([0.5, 0.5], dtype=np.float32)

    result = classifier.predict("甲乙丙丁戊己")

    assert classifier.tokenizer.last_text == "甲乙戊己"
    assert result.labels[0].label == "人工标签甲"


def test_adapter_must_include_classification_head(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        '{"modules_to_save": ["embed_tokens"]}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="classifier head"):
        validate_sequence_classifier_adapter(adapter)


def test_adapter_accepts_qwen_score_head(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"modules_to_save": ["score"]}', encoding="utf-8")
    assert validate_sequence_classifier_adapter(adapter)["modules_to_save"] == ["score"]
