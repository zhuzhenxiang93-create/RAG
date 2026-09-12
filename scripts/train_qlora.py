from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from legalmind.config import load_yaml
from legalmind.data.dataset import MultiLabelCaseDataset, MultiLabelCollator
from legalmind.data.loader import iter_jsonl
from legalmind.data.sampling import select_multilabel_subset, subset_report
from legalmind.models.loading import configure_padding
from legalmind.models.metrics import multilabel_metrics
from legalmind.models.qlora import build_qlora_classifier, trainable_parameter_summary
from legalmind.training.token_bucket import (
    TokenBucketSampler,
    load_token_lengths,
    partial_batch_loss_scale,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_qwen3_4b.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--smoke-samples", type=int)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args()
    model_config = load_yaml(args.model_config)
    training = load_yaml(args.training_config)
    if args.smoke_samples and (args.train_samples or args.validation_samples):
        parser.error("--smoke-samples cannot be combined with explicit subset sizes")
    output_dir = Path(args.output_dir or training["output_dir"])
    if args.smoke_samples:
        output_dir = output_dir.with_name(f"{output_dir.name}_smoke")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_from_checkpoint:
        raise FileExistsError(
            f"Refusing to overwrite existing experiment {output_dir}; use a new output directory"
        )
    started_at = time.time()

    from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        use_fast=True,
    )
    tokenizer.padding_side = "right"
    model = build_qlora_classifier(model_config)
    configure_padding(tokenizer, model)
    print(json.dumps(trainable_parameter_summary(model), indent=2))

    dataset_dir = Path(training["dataset_dir"])
    train_records = list(iter_jsonl(dataset_dir / "train.jsonl"))
    eval_records = list(iter_jsonl(dataset_dir / "validation.jsonl"))
    num_labels = int(model_config["num_labels"])
    seed = int(training.get("seed", 42))
    sampling = {}
    if args.smoke_samples:
        train_records, sampling["train"] = select_multilabel_subset(
            train_records, args.smoke_samples, num_labels, seed=seed
        )
        eval_records, sampling["validation"] = select_multilabel_subset(
            eval_records, max(32, args.smoke_samples // 10), num_labels, seed=seed
        )
    else:
        if args.train_samples:
            train_records, sampling["train"] = select_multilabel_subset(
                train_records, args.train_samples, num_labels, seed=seed
            )
        if args.validation_samples:
            eval_records, sampling["validation"] = select_multilabel_subset(
                eval_records, args.validation_samples, num_labels, seed=seed
            )
    sampling.setdefault("train", subset_report(train_records, num_labels))
    sampling.setdefault("validation", subset_report(eval_records, num_labels))
    max_length = int(training.get("max_length", 1024))
    truncation_strategy = str(training.get("truncation_strategy", "head_tail"))
    train_dataset = MultiLabelCaseDataset(
        train_records, tokenizer, num_labels, max_length, truncation_strategy
    )
    eval_dataset = MultiLabelCaseDataset(
        eval_records, tokenizer, num_labels, max_length, truncation_strategy
    )
    collator = MultiLabelCollator(
        tokenizer, pad_to_multiple_of=int(training.get("pad_to_multiple_of", 8))
    )
    use_length_grouping = bool(training.get("group_by_length", True))
    token_length_cache = training.get("token_length_cache")
    bucket_boundaries = [
        int(value)
        for value in training.get("length_bucket_boundaries", [256, 512, 1024, 2048])
    ]
    full_train_selected = not args.smoke_samples and not args.train_samples
    if use_length_grouping and token_length_cache and full_train_selected:
        training_lengths = load_token_lengths(token_length_cache, len(train_records))
    elif use_length_grouping:
        training_lengths = [
            len(tokenizer(row["fact"], add_special_tokens=True)["input_ids"])
            for row in train_records
        ]
    else:
        training_lengths = []

    class LegalLengthGroupedTrainer(Trainer):
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            actual_batch_size = int(inputs["labels"].shape[0])
            result = super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
            scale = partial_batch_loss_scale(
                actual_batch_size, int(training["per_device_train_batch_size"])
            )
            if return_outputs:
                loss, outputs = result
                return loss * scale, outputs
            return result * scale

        def _get_train_sampler(self, train_dataset=None):
            if not use_length_grouping:
                return super()._get_train_sampler(train_dataset)
            return TokenBucketSampler(
                lengths=training_lengths,
                batch_size=int(training["per_device_train_batch_size"]),
                boundaries=bucket_boundaries,
                seed=seed,
            )

    def compute_metrics(output):
        logits = (
            output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
        )
        return multilabel_metrics(np.asarray(logits), np.asarray(output.label_ids))

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        warmup_ratio=float(training["warmup_ratio"]),
        weight_decay=float(training["weight_decay"]),
        logging_steps=int(training["logging_steps"]),
        eval_strategy="epoch"
        if sampling["train"]["rows"] < sampling["train"]["source_rows"]
        else "steps",
        eval_steps=int(training["eval_steps"]),
        save_strategy="epoch"
        if sampling["train"]["rows"] < sampling["train"]["source_rows"]
        else "steps",
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        bf16=bool(training.get("bf16", True)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        report_to="none",
        seed=int(training.get("seed", 42)),
        remove_unused_columns=False,
    )
    trainer = LegalLengthGroupedTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    import torch

    run_manifest = {
        "model": model_config["name_or_path"],
        "adapter_method": (
            "qlora"
            if bool(model_config.get("quantization", {}).get("load_in_4bit", True))
            else "lora"
        ),
        "quantization": model_config.get("quantization", {}),
        "output_dir": str(output_dir),
        "smoke_samples": args.smoke_samples,
        "requested_train_samples": args.train_samples,
        "requested_validation_samples": args.validation_samples,
        "train_rows": len(train_records),
        "validation_rows": len(eval_records),
        "num_labels": num_labels,
        "max_length": max_length,
        "truncation_strategy": truncation_strategy,
        "dynamic_padding": True,
        "pad_to_multiple_of": int(training.get("pad_to_multiple_of", 8)),
        "group_by_length": bool(training.get("group_by_length", True)),
        "length_grouping_strategy": "token_bucket" if use_length_grouping else "disabled",
        "length_bucket_boundaries": bucket_boundaries if use_length_grouping else [],
        "token_length_cache": str(token_length_cache) if token_length_cache else None,
        "effective_tokens": collator.stats.effective_tokens,
        "padding_tokens": collator.stats.padding_tokens,
        "padding_ratio": collator.stats.padding_ratio,
        "data_sha256": {
            "train": sha256_file(dataset_dir / "train.jsonl"),
            "validation": sha256_file(dataset_dir / "validation.jsonl"),
            "label_mapping": sha256_file(dataset_dir / "label_mapping.json"),
        },
        "elapsed_seconds": time.time() - started_at,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        ),
        "best_metric": trainer.state.best_metric,
        "train_metrics": train_result.metrics,
        "final_validation_metrics": final_eval_metrics,
        "sampling": sampling,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
