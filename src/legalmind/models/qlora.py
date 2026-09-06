from __future__ import annotations

from typing import Any


def build_qlora_classifier(model_config: dict[str, Any]):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    quant = model_config.get("quantization", {})
    dtype_name = quant.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = getattr(torch, dtype_name)
    load_in_4bit = bool(quant.get("load_in_4bit", True))
    loading_options: dict[str, Any] = {"dtype": compute_dtype}
    if load_in_4bit:
        loading_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["name_or_path"],
        num_labels=int(model_config["num_labels"]),
        problem_type="multi_label_classification",
        device_map="auto",
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        **loading_options,
    )
    model.config.pad_token_id = model.config.pad_token_id or model.config.eos_token_id
    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    else:
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
