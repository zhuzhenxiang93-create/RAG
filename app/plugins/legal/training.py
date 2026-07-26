"""Reusable LoRA training configuration without import-time model loading."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class LegalTrainingConfig:
    base_model: str
    output_dir: str
    num_labels: int
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = None
    learning_rate: float = 2e-4
    batch_size: int = 4
    epochs: int = 3
    max_length: int = 512
    seed: int = 42

    def __post_init__(self) -> None:
        if self.target_modules is None:
            object.__setattr__(self, "target_modules", ["q_proj", "v_proj"])
        if self.num_labels < 2:
            raise ValueError("num_labels must be at least 2")


def build_lora_config(config: LegalTrainingConfig):
    """Build PEFT config only when training dependencies are installed."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
        inference_mode=False,
    )
