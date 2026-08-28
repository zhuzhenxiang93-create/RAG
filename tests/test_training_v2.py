from __future__ import annotations

import torch

from legalmind.training.classifier_collator import DynamicMultiLabelCollator
from legalmind.training.long_text import head_tail_truncate, sentence_fact_select
from legalmind.training.samplers import LengthBucketBatchSampler


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, values, skip_special_tokens=True):
        return "".join(chr(value) for value in values)

    def pad(self, features, padding=True, pad_to_multiple_of=None, return_tensors="pt"):
        longest = max(len(row["input_ids"]) for row in features)
        if pad_to_multiple_of:
            longest = (
                (longest + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of
        ids, masks = [], []
        for row in features:
            length = len(row["input_ids"])
            ids.append(row["input_ids"] + [0] * (longest - length))
            masks.append([1] * length + [0] * (longest - length))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


def test_dynamic_padding_tracks_padding_ratio() -> None:
    collator = DynamicMultiLabelCollator(FakeTokenizer(), pad_to_multiple_of=4)
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": torch.tensor([1.0])},
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": torch.tensor([0.0])},
        ]
    )
    assert batch["input_ids"].shape == (2, 4)
    assert collator.stats.effective_tokens == 5
    assert collator.stats.padding_tokens == 3


def test_length_bucket_sampler_is_reproducible() -> None:
    sampler = LengthBucketBatchSampler([9, 1, 8, 2, 7, 3], batch_size=2, seed=7)
    assert list(sampler) == list(LengthBucketBatchSampler([9, 1, 8, 2, 7, 3], 2, 7))


def test_head_tail_preserves_both_boundaries() -> None:
    value = head_tail_truncate(FakeTokenizer(), "ABCDEFGHIJ", 6)
    assert value.startswith("ABCD") and value.endswith("IJ")


def test_sentence_selector_keeps_key_fact() -> None:
    text = "天气正常。被告人持刀致人轻伤。随后离开现场。案件材料很多很多。"
    value = sentence_fact_select(FakeTokenizer(), text, 18)
    assert "持刀" in value
