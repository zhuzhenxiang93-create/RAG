from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def run_baselines(data_dir: Path, output_dir: Path, max_train: int | None = None) -> dict:
    started = time.perf_counter()
    train = load_rows(data_dir / "train.jsonl", max_train)
    validation = load_rows(data_dir / "validation.jsonl")
    mapping = json.loads((data_dir / "label_mapping.json").read_text(encoding="utf-8"))
    classes = list(range(len(mapping)))
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])
    y_train = mlb.transform([row["accusation_ids"] for row in train])
    y_validation = mlb.transform([row["accusation_ids"] for row in validation])

    label_counts = Counter(value for row in train for value in row["accusation_ids"])
    majority_label = label_counts.most_common(1)[0][0]
    majority = np.zeros_like(y_validation)
    majority[:, majority_label] = 1
    majority_f1 = float(f1_score(y_validation, majority, average="micro", zero_division=0))

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=60_000,
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_x = vectorizer.fit_transform([row["fact"] for row in train])
    validation_x = vectorizer.transform([row["fact"] for row in validation])
    classifier = OneVsRestClassifier(
        SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=20, random_state=42, n_jobs=1),
        n_jobs=-1,
    )
    classifier.fit(train_x, y_train)
    probabilities = classifier.predict_proba(validation_x)
    tfidf_predictions = probabilities >= 0.5
    empty = np.where(tfidf_predictions.sum(axis=1) == 0)[0]
    tfidf_predictions[empty, probabilities[empty].argmax(axis=1)] = True
    tfidf_f1 = float(f1_score(y_validation, tfidf_predictions, average="micro", zero_division=0))
    result = {
        "scope": "validation",
        "primary_metric": "micro_f1",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "num_labels": len(mapping),
        "majority": {"micro_f1": majority_f1, "predicted_label_id": majority_label},
        "tfidf_sgd_ovr": {
            "micro_f1": tfidf_f1,
            "features": int(train_x.shape[1]),
            "analyzer": "character_2_4_grams",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed_v2")
    parser.add_argument("--output-dir", default="artifacts/experiments/baselines_v2")
    parser.add_argument("--max-train", type=int)
    args = parser.parse_args()
    result = run_baselines(Path(args.data_dir), Path(args.output_dir), args.max_train)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
