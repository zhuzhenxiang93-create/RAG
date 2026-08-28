from __future__ import annotations

from typing import Any

import torch

from legalmind.training.classifier_collator import DynamicMultiLabelCollator
from legalmind.training.long_text import truncate_fact


class MultiLabelCaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        num_labels: int,
        max_length: int,
        truncation_strategy: str = "head_tail",
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        fact = truncate_fact(
            self.tokenizer,
            record["fact"],
            self.max_length,
            strategy=self.truncation_strategy,
        )
        encoded = self.tokenizer(
            fact,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        labels = torch.zeros(self.num_labels, dtype=torch.float32)
        for label_id in record["accusation_ids"]:
            if 0 <= int(label_id) < self.num_labels:
                labels[int(label_id)] = 1.0
        encoded["labels"] = labels
        return encoded


MultiLabelCollator = DynamicMultiLabelCollator
