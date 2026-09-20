from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from legalmind.config import load_yaml
from legalmind.data.dataset import MultiLabelCaseDataset, MultiLabelCollator
from legalmind.data.labels import load_label_mapping
from legalmind.data.loader import iter_jsonl
from legalmind.data.sampling import select_multilabel_subset
from legalmind.models.metrics import binarize, multilabel_report, sigmoid, tune_thresholds
from legalmind.models.peft_classifier import load_adapter_classifier


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_thresholds(path: Path, num_labels: int) -> np.ndarray:
    values = json.loads(path.read_text(encoding="utf-8"))
    if len(values) != num_labels or set(map(int, values)) != set(range(num_labels)):
        raise ValueError("threshold IDs must exactly match label IDs")
    return np.asarray([float(values[str(index)]) for index in range(num_labels)], dtype=np.float32)


def predict(model, dataset, collator, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        labels.append(batch.pop("labels").numpy())
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.inference_mode():
            logits.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels)


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.quantile(values, quantile))
        for name, quantile in {
            "min": 0.0,
            "p25": 0.25,
            "p50": 0.5,
            "p75": 0.75,
            "p95": 0.95,
            "max": 1.0,
        }.items()
    }


def probability_audit(logits: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> dict:
    probabilities = sigmoid(logits)
    predictions = binarize(probabilities, thresholds, ensure_one=False)
    positives = probabilities[labels.astype(bool)]
    negatives = probabilities[~labels.astype(bool)]
    return {
        "row_max_probability": quantiles(probabilities.max(axis=1)),
        "true_label_probability": quantiles(positives),
        "negative_label_probability": quantiles(negatives),
        "predicted_cardinality": quantiles(predictions.sum(axis=1)),
        "abstained_rows": int((predictions.sum(axis=1) == 0).sum()),
        "rows": int(labels.shape[0]),
    }


def identity_audit(args, training: dict, label_to_id: dict[str, int]) -> dict:
    dataset_dir = Path(training["dataset_dir"])
    validation_path = dataset_dir / "validation.jsonl"
    records = list(iter_jsonl(validation_path))
    current_subset_report = None
    if len(records) >= 2000:
        _, current_subset_report = select_multilabel_subset(
            records, 2000, len(label_to_id), seed=int(training.get("seed", 42))
        )
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    old_report_path = Path(args.historical_report)
    old_report = (
        json.loads(old_report_path.read_text(encoding="utf-8")) if old_report_path.exists() else {}
    )
    historical_validation = old_report.get("sampling", {}).get("validation", {})
    actual_hash = sha256(validation_path)
    manifest_hash = manifest.get("output_sha256", {}).get("validation.jsonl")
    historical_selected_hash = historical_validation.get("selected_indices_sha256")
    current_selected_hash = (
        current_subset_report.get("selected_indices_sha256") if current_subset_report else None
    )
    return {
        "adapter": {
            "directory": args.adapter,
            "config_sha256": sha256(Path(args.adapter) / "adapter_config.json"),
            "weights_sha256": sha256(Path(args.adapter) / "adapter_model.safetensors"),
        },
        "label_mapping": {
            "path": args.label_mapping,
            "sha256": sha256(Path(args.label_mapping)),
            "labels": len(label_to_id),
            "contiguous_ids": set(label_to_id.values()) == set(range(len(label_to_id))),
        },
        "thresholds": {
            "path": args.thresholds,
            "sha256": sha256(Path(args.thresholds)),
            "count": len(json.loads(Path(args.thresholds).read_text(encoding="utf-8"))),
        },
        "dataset": {
            "directory": str(dataset_dir),
            "version": manifest.get("dataset_version"),
            "validation_rows_current": len(records),
            "validation_rows_historical_report": historical_validation.get("source_rows"),
            "validation_sha256_actual": actual_hash,
            "validation_sha256_manifest": manifest_hash,
            "manifest_matches_actual": manifest_hash == actual_hash,
            "historical_selected_indices_sha256": historical_selected_hash,
            "current_selected_indices_sha256": current_selected_hash,
            "historical_subset_reproducible": historical_selected_hash == current_selected_hash,
        },
        "preprocessing": {
            "max_length": int(training["max_length"]),
            "truncation_strategy": training.get("truncation_strategy", "head_tail"),
            "seed": int(training.get("seed", 42)),
        },
        "identity_gate_passed": (
            manifest_hash == actual_hash
            and historical_validation.get("source_rows") == len(records)
            and historical_selected_hash == current_selected_hash
        ),
    }


def regression_records(path: Path, label_to_id: dict[str, int]) -> tuple[list[dict], list[dict]]:
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    records = []
    for row in raw:
        unknown = (set(row["expected_accusations"]) | set(row["forbidden_accusations"])) - set(
            label_to_id
        )
        if unknown:
            raise ValueError(f"unknown regression labels: {sorted(unknown)}")
        records.append(
            {
                "case_id": row["case_id"],
                "fact": row["fact"],
                "accusation_ids": [label_to_id[label] for label in row["expected_accusations"]],
            }
        )
    return raw, records


def regression_report(
    raw: list[dict], logits: np.ndarray, thresholds: np.ndarray, id_to_label: dict[int, str]
) -> dict:
    probabilities = sigmoid(logits)
    threshold_predictions = binarize(probabilities, thresholds, ensure_one=False)
    rows = []
    for index, source in enumerate(raw):
        predicted_ids = np.flatnonzero(threshold_predictions[index]).tolist()
        ranked_ids = np.argsort(-probabilities[index])[:5].tolist()
        predicted = {id_to_label[label_id] for label_id in predicted_ids}
        top5 = [id_to_label[label_id] for label_id in ranked_ids]
        expected = set(source["expected_accusations"])
        forbidden = set(source["forbidden_accusations"])
        threshold_pass = (
            not predicted if source["expect_abstain"] else expected.issubset(predicted)
        ) and predicted.isdisjoint(forbidden)
        rows.append(
            {
                "case_id": source["case_id"],
                "category": source["category"],
                "expected": sorted(expected),
                "forbidden": sorted(forbidden),
                "threshold_predictions": sorted(predicted),
                "top5": top5,
                "expected_in_top5": bool(expected) and expected.issubset(set(top5)),
                "abstained": not predicted,
                "pass": threshold_pass,
                "max_probability": float(probabilities[index].max()),
            }
        )
    categories: dict[str, dict[str, int | float]] = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        categories[category] = {
            "rows": len(selected),
            "passed": sum(bool(row["pass"]) for row in selected),
            "pass_rate": sum(bool(row["pass"]) for row in selected) / len(selected),
        }
    return {
        "rows": len(rows),
        "passed": sum(bool(row["pass"]) for row in rows),
        "pass_rate": sum(bool(row["pass"]) for row in rows) / len(rows),
        "expected_in_top5_rate": sum(
            bool(row["expected_in_top5"]) for row in rows if row["expected"]
        )
        / max(1, sum(bool(row["expected"]) for row in rows)),
        "abstention_cases": sum(bool(row["abstained"]) for row in rows),
        "categories": categories,
        "cases": rows,
    }


def load_model(model_config: dict, adapter: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"], trust_remote_code=True
    )
    return tokenizer, load_adapter_classifier(model_config, adapter)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only classifier regression diagnosis")
    parser.add_argument("--model-config", default="configs/model_qwen3_4b.yaml")
    parser.add_argument(
        "--training-config", default="configs/classification/qwen3_4b_qlora_10k.yaml"
    )
    parser.add_argument("--adapter", default="artifacts/models/qwen3_4b_qlora_10k")
    parser.add_argument(
        "--thresholds", default="artifacts/results/classifier_10k_2k/thresholds.json"
    )
    parser.add_argument("--label-mapping", default="data/processed_v2/label_mapping.json")
    parser.add_argument(
        "--historical-report", default="artifacts/results/classifier_10k_2k/evaluation_report.json"
    )
    parser.add_argument(
        "--historical-validation-logits",
        default="artifacts/results/classifier_10k_2k/validation_logits.npy",
    )
    parser.add_argument(
        "--historical-validation-labels",
        default="artifacts/results/classifier_10k_2k/validation_labels.npy",
    )
    parser.add_argument("--regression-set", default="tests/fixtures/classifier_regression_30.jsonl")
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument(
        "--output", default="reports/classification/classifier_regression_diagnosis.json"
    )
    args = parser.parse_args()

    started = time.time()
    model_config = load_yaml(args.model_config)
    training = load_yaml(args.training_config)
    label_to_id = load_label_mapping(args.label_mapping)
    id_to_label = {label_id: label for label, label_id in label_to_id.items()}
    num_labels = int(model_config["num_labels"])
    if num_labels != len(label_to_id):
        raise ValueError("model label count and mapping size differ")
    thresholds = load_thresholds(Path(args.thresholds), num_labels)
    identity = identity_audit(args, training, label_to_id)

    historical_logits = np.load(args.historical_validation_logits)
    historical_labels = np.load(args.historical_validation_labels)
    historical_replay = multilabel_report(
        historical_logits, historical_labels, thresholds, id_to_label
    )

    validation_records = list(iter_jsonl(Path(training["dataset_dir"]) / "validation.jsonl"))
    if args.max_validation_samples:
        validation_records = validation_records[: args.max_validation_samples]
    raw_regression, model_regression = regression_records(Path(args.regression_set), label_to_id)

    tokenizer, model = load_model(model_config, args.adapter)
    strategy = training.get("truncation_strategy", "head_tail")
    validation_dataset = MultiLabelCaseDataset(
        validation_records,
        tokenizer,
        num_labels,
        int(training["max_length"]),
        truncation_strategy=strategy,
    )
    regression_dataset = MultiLabelCaseDataset(
        model_regression,
        tokenizer,
        num_labels,
        int(training["max_length"]),
        truncation_strategy=strategy,
    )
    collator = MultiLabelCollator(tokenizer)
    batch_size = int(training["per_device_eval_batch_size"])
    validation_logits, validation_labels = predict(model, validation_dataset, collator, batch_size)
    regression_logits, _ = predict(model, regression_dataset, collator, batch_size)
    current_tuned = tune_thresholds(sigmoid(validation_logits), validation_labels)
    report: dict[str, Any] = {
        "status": "completed",
        "scope": "validation_only_test_split_not_read_by_model_evaluation",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "identity": identity,
        "historical_validation_replay": historical_replay,
        "current_validation": {
            "rows": len(validation_records),
            "historical_thresholds": multilabel_report(
                validation_logits, validation_labels, thresholds, id_to_label
            ),
            "fixed_0_5": multilabel_report(validation_logits, validation_labels, 0.5, id_to_label),
            "retuned_in_sample_diagnostic_only": multilabel_report(
                validation_logits, validation_labels, current_tuned, id_to_label
            ),
            "probability_audit": probability_audit(
                validation_logits, validation_labels, thresholds
            ),
        },
        "threshold_audit": {
            "historical": quantiles(thresholds),
            "current_retuned_in_sample_only": quantiles(current_tuned),
            "changed_labels": int((np.abs(current_tuned - thresholds) > 1e-6).sum()),
        },
        "synthetic_regression": regression_report(
            raw_regression, regression_logits, thresholds, id_to_label
        ),
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "The synthetic set is hand-authored for regression and is not a legal benchmark.",
            "Retuned validation metrics are in-sample diagnostics and must not be reported as test performance.",
            "The historical threshold file is not cryptographically linked to the checkpoint manifest.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "identity_gate_passed": identity["identity_gate_passed"],
                "validation_rows": len(validation_records),
                "historical_replay": historical_replay["summary"],
                "current_validation": report["current_validation"]["historical_thresholds"][
                    "summary"
                ],
                "synthetic": {
                    key: report["synthetic_regression"][key]
                    for key in ("rows", "passed", "pass_rate", "expected_in_top5_rate")
                },
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
