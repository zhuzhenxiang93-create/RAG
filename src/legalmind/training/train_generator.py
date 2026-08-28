from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from legalmind.config import load_yaml
from legalmind.training.sft_dataset import SFTCollator, StructuredSFTDataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generation/qwen3_4b_sft_smoke.yaml")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-validation", type=int)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()) and not args.resume_from_checkpoint:
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output}")
    started = time.time()

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(config.get("lora_r", 16)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            target_modules=config.get("target_modules", "all-linear"),
            bias="none",
        ),
    )
    data_dir = Path(config["dataset_dir"])
    train_dataset = StructuredSFTDataset(
        data_dir / "train.jsonl", tokenizer, int(config["max_length"]), args.max_train
    )
    validation_dataset = StructuredSFTDataset(
        data_dir / "validation.jsonl",
        tokenizer,
        int(config["max_length"]),
        args.max_validation,
    )
    arguments = TrainingArguments(
        output_dir=str(output),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=int(config.get("logging_steps", 5)),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=int(config.get("seed", 42)),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=SFTCollator(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    evaluation = trainer.evaluate()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    manifest = {
        "experiment_type": "generation_sft_smoke" if args.max_train else "generation_sft_full",
        "model": config["model"],
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "max_length": int(config["max_length"]),
        "assistant_only_loss": True,
        "quantization": "4bit_nf4_double_quant_bf16",
        "lora": {
            "r": int(config.get("lora_r", 16)),
            "alpha": int(config.get("lora_alpha", 32)),
            "dropout": float(config.get("lora_dropout", 0.05)),
            "target_modules": config.get("target_modules", "all-linear"),
        },
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "train_metrics": result.metrics,
        "validation_metrics": evaluation,
        "data_sha256": {
            "train": sha256_file(data_dir / "train.jsonl"),
            "validation": sha256_file(data_dir / "validation.jsonl"),
        },
        "status": "completed",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
