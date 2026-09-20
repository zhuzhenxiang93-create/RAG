from __future__ import annotations

from typing import Any


def _dtype_and_quantization(model_config: dict[str, Any]):
    import torch
    from transformers import BitsAndBytesConfig

    quant = model_config.get("quantization", {})
    dtype_name = quant.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = getattr(torch, dtype_name)
    load_in_4bit = bool(quant.get("load_in_4bit", False))
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return compute_dtype, load_in_4bit, quantization_config


def load_base_sequence_classifier(model_config: dict[str, Any]):
    from transformers import AutoModelForSequenceClassification

    compute_dtype, load_in_4bit, quantization_config = _dtype_and_quantization(model_config)
    loading_options: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": compute_dtype,
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
    }
    if quantization_config is not None:
        loading_options["quantization_config"] = quantization_config

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["name_or_path"],
        num_labels=int(model_config["num_labels"]),
        problem_type="multi_label_classification",
        **loading_options,
    )
    model.config.pad_token_id = model.config.pad_token_id or model.config.eos_token_id
    return model, load_in_4bit


def build_adapter_classifier(model_config: dict[str, Any]):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model, load_in_4bit = load_base_sequence_classifier(model_config)
    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    else:
        # BF16 LoRA path used on the school A100.
        model.enable_input_require_grads()

    lora = model_config.get("lora", {})
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("lora_alpha", 32)),
        lora_dropout=float(lora.get("lora_dropout", 0.05)),
        target_modules=lora.get("target_modules", "all-linear"),
        modules_to_save=lora.get("modules_to_save", ["score"]),
        bias=lora.get("bias", "none"),
    )
    return get_peft_model(model, peft_config)


def load_adapter_classifier(model_config: dict[str, Any], adapter_path: str):
    from peft import PeftModel

    base, _ = load_base_sequence_classifier(model_config)
    return PeftModel.from_pretrained(base, adapter_path).eval()


def adapter_method(model_config: dict[str, Any]) -> str:
    return (
        "qlora"
        if bool(model_config.get("quantization", {}).get("load_in_4bit", False))
        else "lora"
    )


def trainable_parameter_summary(model) -> dict[str, float | int]:
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }
