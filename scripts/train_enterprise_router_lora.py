"""Fine-tune a causal LM with LoRA for structured enterprise query routing."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.plugins.enterprise_router.data import read_jsonl
from app.plugins.enterprise_router.training import (
    RouterDataCollator,
    RouterTrainingConfig,
    build_lora_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/enterprise_router"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", default="artifacts/enterprise-router-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use QLoRA. Requires a supported CUDA environment and bitsandbytes.",
    )
    parser.add_argument(
        "--smoke-max-steps",
        type=int,
        help="Run only N optimizer steps to validate the training path; not a real experiment.",
    )
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from peft import get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install full dependencies first: pip install -r requirements-full.txt"
        ) from exc

    train_path = args.data_dir / "train.jsonl"
    validation_path = args.data_dir / "validation.jsonl"
    if not train_path.is_file() or not validation_path.is_file():
        raise SystemExit(
            "Prepared train/validation files are missing. Run prepare_enterprise_router.py."
        )
    if (args.data_dir / "benchmark.jsonl").resolve() in {
        train_path.resolve(),
        validation_path.resolve(),
    }:
        raise SystemExit("Benchmark data cannot be used as a training split")

    config = RouterTrainingConfig(
        base_model=args.base_model,
        output_dir=args.output,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        max_length=args.max_length,
        seed=args.seed,
    )
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                __import__("torch").bfloat16 if args.bf16 else __import__("torch").float16
            ),
        )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto" if args.load_in_4bit else None,
    )
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(config))
    model.print_trainable_parameters()

    columns = ["prompt", "target"]
    train_dataset = Dataset.from_list(
        [{key: row[key] for key in columns} for row in read_jsonl(train_path)]
    )
    validation_dataset = Dataset.from_list(
        [{key: row[key] for key in columns} for row in read_jsonl(validation_path)]
    )
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(output),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        max_steps=args.smoke_max_steps or -1,
        eval_strategy="steps" if args.smoke_max_steps else "epoch",
        eval_steps=1 if args.smoke_max_steps else None,
        save_strategy="no" if args.smoke_max_steps else "epoch",
        logging_steps=1 if args.smoke_max_steps else 20,
        seed=config.seed,
        data_seed=config.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=RouterDataCollator(tokenizer, config.max_length),
    )
    train_result = trainer.train()
    adapter_dir = output / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = {
        **train_result.metrics,
        **trainer.evaluate(),
        "experiment_kind": "smoke_test" if args.smoke_max_steps else "full_training",
        "base_model": config.base_model,
        "seed": config.seed,
        "rank": config.rank,
        "alpha": config.alpha,
        "target_modules": config.target_modules,
        "load_in_4bit": args.load_in_4bit,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
    }
    (output / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

