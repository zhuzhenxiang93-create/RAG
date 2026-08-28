from __future__ import annotations

import argparse
import heapq
import json
import time
from pathlib import Path

import torch

from legalmind.config import load_yaml
from legalmind.data.dataset import MultiLabelCaseDataset, MultiLabelCollator
from legalmind.data.loader import iter_jsonl
from legalmind.models.loading import configure_padding
from legalmind.models.qlora import build_qlora_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_qwen3_4b.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--output", default="artifacts/results/batch_stress_b4.json")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    args = parser.parse_args()
    model_config = load_yaml(args.model_config)
    training = load_yaml(args.training_config)
    batch_size = args.batch_size or int(training["per_device_train_batch_size"])
    max_length = int(training["max_length"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        use_fast=True,
    )
    tokenizer.padding_side = "right"
    model = build_qlora_classifier(model_config)
    configure_padding(tokenizer, model)
    model.train()

    train_path = Path(training["dataset_dir"]) / "train.jsonl"
    longest = heapq.nlargest(batch_size, iter_jsonl(train_path), key=lambda row: len(row["fact"]))
    dataset = MultiLabelCaseDataset(
        longest,
        tokenizer,
        int(model_config["num_labels"]),
        max_length,
    )
    features = [dataset[index] for index in range(len(dataset))]
    collator = MultiLabelCollator(
        tokenizer, pad_to_multiple_of=int(training.get("pad_to_multiple_of", 8))
    )
    batch = collator(features)
    batch = {key: value.to(model.device) for key, value in batch.items()}

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.time()
    status = "completed"
    loss = None
    error = None
    try:
        model_output = model(**batch)
        model_output.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        loss = float(model_output.loss.detach().cpu())
    except torch.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc).splitlines()[0]
        torch.cuda.empty_cache()
    result = {
        "status": status,
        "batch_size": batch_size,
        "gradient_accumulation_steps": max(1, args.effective_batch_size // batch_size),
        "effective_batch_size": args.effective_batch_size,
        "sequence_length": int(batch["input_ids"].shape[1]),
        "effective_tokens": collator.stats.effective_tokens,
        "padding_tokens": collator.stats.padding_tokens,
        "padding_ratio": collator.stats.padding_ratio,
        "loss": loss,
        "error": error,
        "step_seconds": time.time() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "gpu_name": torch.cuda.get_device_name(),
    }
    result_text = json.dumps(result, ensure_ascii=False, indent=2)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result_text, encoding="utf-8")
    print(result_text)


if __name__ == "__main__":
    main()
