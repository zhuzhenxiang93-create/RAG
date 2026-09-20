"""Backward-compatible PEFT classifier imports.

Historical experiments used this module name for QLoRA. New school training uses
BF16 LoRA (load_in_4bit=false) through :mod:`legalmind.models.peft_classifier`.
"""

from legalmind.models.peft_classifier import (
    adapter_method,
    build_adapter_classifier,
    load_adapter_classifier,
    trainable_parameter_summary,
)


def build_qlora_classifier(model_config):
    """Compatibility alias; dispatches to LoRA or QLoRA from model_config."""
    return build_adapter_classifier(model_config)


__all__ = [
    "adapter_method",
    "build_adapter_classifier",
    "build_qlora_classifier",
    "load_adapter_classifier",
    "trainable_parameter_summary",
]
