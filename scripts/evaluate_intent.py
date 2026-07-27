"""Evaluate an intent adapter on a prepared MASSIVE test split."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings
from app.evaluation.classification import classification_metrics
from app.plugins.intent.assets import load_taxonomy
from app.plugins.intent.classifier import IntentClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/intent/massive")
    parser.add_argument("--locale", default="zh-CN")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--taxonomy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="artifacts/intent-evaluation.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    taxonomy_path = Path(args.taxonomy) if args.taxonomy else data_dir / "taxonomy.json"
    intents, _ = load_taxonomy(taxonomy_path)
    settings = Settings(
        project_root=Path.cwd(),
        intent_enabled=True,
        intent_backend="lora",
        intent_base_model=args.base_model,
        intent_adapter_path=args.adapter,
        intent_labels_path=str(taxonomy_path),
        intent_device=args.device,
    )
    classifier = IntentClassifier(settings)
    gold = []
    predicted = []
    confidences = []
    predictions = []
    test_path = data_dir / f"{args.locale}.test.jsonl"
    with test_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if args.limit is not None and index >= args.limit:
                break
            row = json.loads(line)
            response = classifier.predict(row["text"], top_k=3)
            top = response.predictions[0]
            gold.append(intents[row["intent"]])
            predicted.append(top.label_id)
            confidences.append(top.probability)
            predictions.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "gold_intent": row["intent"],
                    "predicted_intent": top.intent,
                    "confidence": top.probability,
                    "correct": row["intent"] == top.intent,
                }
            )
    result = classification_metrics(
        gold,
        predicted,
        confidences,
        label_count=len(intents),
    )
    result.update(
        {
            "dataset": "AmazonScience/massive",
            "locale": args.locale,
            "base_model": args.base_model,
            "adapter": args.adapter,
            "limited": args.limit is not None,
            "predictions": predictions,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in ("sample_count", "accuracy", "macro_f1", "ece")}, indent=2))


if __name__ == "__main__":
    main()
