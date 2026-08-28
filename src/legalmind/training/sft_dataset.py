from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class StructuredSFTDataset(torch.utils.data.Dataset):
    def __init__(
        self, path: str | Path, tokenizer: Any, max_length: int, max_rows: int | None = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.rows.append(json.loads(line))
                    if max_rows and len(self.rows) >= max_rows:
                        break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        messages = self.rows[index]["messages"]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        if len(full_ids) > self.max_length:
            overflow = len(full_ids) - self.max_length
            full_ids = full_ids[overflow:]
            prompt_length = max(0, len(prompt_ids) - overflow)
        else:
            prompt_length = len(prompt_ids)
        labels = [-100] * prompt_length + full_ids[prompt_length:]
        if not any(value != -100 for value in labels):
            raise ValueError("Assistant mask removed all supervised tokens")
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class SFTCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        labels = [row["labels"] for row in features]
        model_features = [
            {key: value for key, value in row.items() if key != "labels"} for row in features
        ]
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        padded_labels = torch.full_like(batch["input_ids"], -100)
        for row_index, values in enumerate(labels):
            padded_labels[row_index, : len(values)] = torch.tensor(values, dtype=torch.long)
        batch["labels"] = padded_labels
        return batch
