"""Reusable PEFT configuration and prompt-masked causal-LM collation."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RouterTrainingConfig:
    base_model: str
    output_dir: str
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    epochs: float = 3.0
    max_length: int = 512
    seed: int = 42

    def __post_init__(self) -> None:
        if self.target_modules is None:
            object.__setattr__(
                self, "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            )
        if self.rank < 1 or self.alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if self.max_length < 64:
            raise ValueError("max_length is too small for the routing schema")


def build_lora_config(config: RouterTrainingConfig):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
        bias="none",
        inference_mode=False,
    )


class RouterDataCollator:
    """Tokenize prompt+target while masking prompt tokens from the LM loss."""

    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]):
        import torch

        pad_id = self.tokenizer.pad_token_id
        rows = []
        for feature in features:
            prompt_ids = self.tokenizer(
                feature["prompt"],
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            target_ids = self.tokenizer(
                feature["target"] + self.tokenizer.eos_token,
                add_special_tokens=False,
                truncation=True,
                max_length=max(self.max_length - len(prompt_ids), 1),
            )["input_ids"]
            input_ids = (prompt_ids + target_ids)[: self.max_length]
            prompt_length = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_length + input_ids[prompt_length:]
            rows.append((input_ids, labels))
        width = max(len(item[0]) for item in rows)
        input_batch = []
        label_batch = []
        attention_batch = []
        for input_ids, labels in rows:
            padding = width - len(input_ids)
            input_batch.append(input_ids + [pad_id] * padding)
            label_batch.append(labels + [-100] * padding)
            attention_batch.append([1] * len(input_ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_batch, dtype=torch.long),
            "labels": torch.tensor(label_batch, dtype=torch.long),
        }
