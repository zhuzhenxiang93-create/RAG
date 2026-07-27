"""Reusable LoRA sequence-classification configuration."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class IntentTrainingConfig:
    base_model: str
    output_dir: str
    num_labels: int
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    learning_rate: float = 2e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: float = 3.0
    max_length: int = 256
    seed: int = 42

    def __post_init__(self) -> None:
        if self.target_modules is None:
            object.__setattr__(self, "target_modules", ["q_proj", "v_proj"])
        if self.num_labels < 2:
            raise ValueError("num_labels must be at least 2")
        if self.rank < 1 or self.alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")


def build_lora_config(config: IntentTrainingConfig):
    """Build a PEFT config and persist the classifier head with the adapter."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
        modules_to_save=["score"],
        inference_mode=False,
    )

