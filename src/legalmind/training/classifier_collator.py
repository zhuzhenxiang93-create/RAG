from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PaddingStats:
    effective_tokens: int = 0
    padding_tokens: int = 0
    batches: int = 0

    @property
    def padding_ratio(self) -> float:
        total = self.effective_tokens + self.padding_tokens
        return self.padding_tokens / total if total else 0.0


@dataclass
class DynamicMultiLabelCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8
    stats: PaddingStats = field(default_factory=PaddingStats)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.stack([feature["labels"] for feature in features])
        model_features = [
            {key: value for key, value in row.items() if key != "labels"} for row in features
        ]
        effective = sum(len(row["input_ids"]) for row in model_features)
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        allocated = int(batch["attention_mask"].numel())
        self.stats.effective_tokens += effective
        self.stats.padding_tokens += allocated - effective
        self.stats.batches += 1
        batch["labels"] = labels
        return batch
