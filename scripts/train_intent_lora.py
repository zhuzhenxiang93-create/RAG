"""Train a Qwen sequence classifier with LoRA on normalized MASSIVE data."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.classification import macro_f1_from_logits
from app.plugins.intent.assets import load_taxonomy
from app.plugins.intent.training import IntentTrainingConfig, build_lora_config


def _read_jsonl(paths):
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/intent/massive")
    parser.add_argument("--locale", action="append", default=None)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output", default="artifacts/intent-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
        from datasets import Dataset
        from peft import get_peft_model
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install Full dependencies first: pip install -r requirements-full.txt"
        ) from exc

    data_dir = Path(args.data_dir)
    locales = args.locale or ["zh-CN"]
    intents, intent_to_domain = load_taxonomy(data_dir / "taxonomy.json")
    split_aliases = {"train": "train", "validation": "validation", "test": "test"}
    rows = {}
    for name, file_split in split_aliases.items():
        paths = [data_dir / f"{locale}.{file_split}.jsonl" for locale in locales]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise SystemExit("Missing prepared split(s): " + ", ".join(missing))
        rows[name] = _read_jsonl(paths)

    config = IntentTrainingConfig(
        base_model=args.base_model,
        output_dir=args.output,
        num_labels=len(intents),
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
    id2label = {identifier: label for label, identifier in intents.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=config.num_labels,
        id2label=id2label,
        label2id=intents,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(model, build_lora_config(config))
    model.print_trainable_parameters()

    def build_dataset(split_rows):
        dataset = Dataset.from_list(
            [
                {"text": row["text"], "labels": intents[row["intent"]]}
                for row in split_rows
            ]
        )
        return dataset.map(
            lambda batch: tokenizer(
                batch["text"], truncation=True, max_length=config.max_length
            ),
            batched=True,
            remove_columns=["text"],
        )

    train_dataset = build_dataset(rows["train"])
    validation_dataset = build_dataset(rows["validation"])

    def compute_metrics(evaluation):
        logits, labels = evaluation
        return macro_f1_from_logits(np.asarray(logits), np.asarray(labels).tolist())

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(output),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=config.seed,
        data_seed=config.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(str(output / "adapter"))
    taxonomy_payload = {
        "dataset": "AmazonScience/massive",
        "locales": locales,
        "intents": intents,
        "intent_to_domain": intent_to_domain,
    }
    (output / "taxonomy.json").write_text(
        json.dumps(taxonomy_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics = {
        **train_result.metrics,
        **trainer.evaluate(),
        "base_model": config.base_model,
        "locales": locales,
        "seed": config.seed,
        "rank": config.rank,
        "alpha": config.alpha,
        "target_modules": config.target_modules,
    }
    (output / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
