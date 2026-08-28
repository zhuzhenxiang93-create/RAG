from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from legalmind.config import load_yaml
from legalmind.data.dataset import MultiLabelCaseDataset, MultiLabelCollator
from legalmind.data.labels import load_label_mapping
from legalmind.data.loader import iter_jsonl
from legalmind.data.sampling import select_multilabel_subset
from legalmind.models.loading import configure_padding
from legalmind.models.metrics import multilabel_report, sigmoid, tune_thresholds


def predict(model, dataset, collator, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    logits, labels = [], []
    model.eval()
    for batch in loader:
        labels.append(batch.pop("labels").numpy())
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.inference_mode():
            logits.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_qwen3_4b.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--adapter")
    parser.add_argument("--output-dir", default="artifacts/results/classifier")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--label-mapping", default="data/manifests/label_mapping.json")
    args = parser.parse_args()
    model_config = load_yaml(args.model_config)
    training = load_yaml(args.training_config)

    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

    dtype = getattr(torch, model_config["quantization"].get("bnb_4bit_compute_dtype", "bfloat16"))
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_config["quantization"].get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    base_name = model_config["name_or_path"]
    adapter = args.adapter or training["output_dir"]
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=int(model_config["num_labels"]),
        problem_type="multi_label_classification",
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    configure_padding(tokenizer, base)
    model = PeftModel.from_pretrained(base, adapter)
    dataset_dir = Path(training["dataset_dir"])
    datasets = {}
    sampling = {}
    num_labels = int(model_config["num_labels"])
    seed = int(training.get("seed", 42))
    for split in ("validation", "test"):
        records = list(iter_jsonl(dataset_dir / f"{split}.jsonl"))
        if args.max_samples:
            records, sampling[split] = select_multilabel_subset(
                records, args.max_samples, num_labels, seed=seed
            )
        datasets[split] = MultiLabelCaseDataset(
            records,
            tokenizer,
            num_labels,
            int(training["max_length"]),
        )
    collator = MultiLabelCollator(tokenizer)
    validation_logits, validation_labels = predict(
        model, datasets["validation"], collator, int(training["per_device_eval_batch_size"])
    )
    test_logits, test_labels = predict(
        model, datasets["test"], collator, int(training["per_device_eval_batch_size"])
    )
    thresholds = tune_thresholds(sigmoid(validation_logits), validation_labels)
    label_to_id = load_label_mapping(args.label_mapping)
    report = multilabel_report(
        test_logits,
        test_labels,
        thresholds,
        label_names={label_id: label for label, label_id in label_to_id.items()},
    )
    metrics = report["summary"]
    report["sampling"] = sampling
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "validation_logits": validation_logits,
        "validation_labels": validation_labels,
        "test_logits": test_logits,
        "test_labels": test_labels,
    }.items():
        np.save(output / f"{name}.npy", value)
    (output / "thresholds.json").write_text(
        json.dumps({str(i): float(v) for i, v in enumerate(thresholds)}, indent=2),
        encoding="utf-8",
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
