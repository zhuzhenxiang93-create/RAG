from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from legalmind.models.metrics import multilabel_report, sigmoid, tune_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cached validation/test logits without repeating model inference."
    )
    parser.add_argument("--validation-logits", required=True)
    parser.add_argument("--validation-labels", required=True)
    parser.add_argument("--test-logits", required=True)
    parser.add_argument("--test-labels", required=True)
    parser.add_argument("--output-dir", default="artifacts/results/classifier")
    args = parser.parse_args()
    validation_logits = np.load(args.validation_logits)
    validation_labels = np.load(args.validation_labels)
    test_logits = np.load(args.test_logits)
    test_labels = np.load(args.test_labels)
    thresholds = tune_thresholds(sigmoid(validation_logits), validation_labels)
    report = multilabel_report(test_logits, test_labels, thresholds)
    metrics = report["summary"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "thresholds.json").write_text(
        json.dumps({str(i): float(value) for i, value in enumerate(thresholds)}, indent=2),
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
