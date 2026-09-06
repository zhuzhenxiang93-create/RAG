from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor

from legalmind.sentencing.labels import (
    FINE_BINS,
    IMPRISONMENT_BINS,
    bucket_range,
    labels_from_row,
    targets_from_labels,
)


def model_text(fact: str, accusations: list[str] | None = None) -> str:
    prefix = f"已知罪名:{'、'.join(accusations)}。" if accusations else "罪名未知。"
    return prefix + fact.strip()


def _training_text(row: dict) -> str:
    accusations = labels_from_row(row).get("accusations", [])
    # Deterministic accusation dropout lets one model serve both optional-input modes.
    digest = hashlib.sha1(row["case_id"].encode("utf-8")).digest()[0]
    return model_text(row["fact"], accusations if digest % 2 else None)


class SentencingBaseline:
    """Auditable hierarchical baseline; no generative model or label-at-inference shortcut."""

    def __init__(self, random_state: int = 42, n_features: int = 2**18):
        self.random_state = random_state
        self.vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=False,
        )
        classifier = {
            "loss": "log_loss",
            "class_weight": "balanced",
            "max_iter": 30,
            "tol": 1e-3,
            "random_state": random_state,
            "n_jobs": -1,
        }
        self.sentence_type_model = SGDClassifier(**classifier)
        self.imprisonment_bucket_model = SGDClassifier(**classifier)
        self.fine_imposed_model = SGDClassifier(**classifier)
        self.fine_bucket_model = SGDClassifier(**classifier)
        self.imprisonment_regressor = SGDRegressor(
            loss="huber", max_iter=40, tol=1e-3, random_state=random_state
        )
        self.fine_regressor = SGDRegressor(
            loss="huber", max_iter=40, tol=1e-3, random_state=random_state
        )
        self.metadata: dict = {}

    def fit(self, rows: list[dict]) -> dict:
        targets = [targets_from_labels(labels_from_row(row)) for row in rows]
        texts = [_training_text(row) for row in rows]
        matrix = self.vectorizer.transform(texts)
        type_indices = [
            i
            for i, target in enumerate(targets)
            if target.sentence_type != "unknown"
            and rows[i].get("task_masks", {}).get("sentencing_train", True)
        ]
        self.sentence_type_model.fit(
            matrix[type_indices], [targets[i].sentence_type for i in type_indices]
        )

        month_indices = [
            i
            for i, target in enumerate(targets)
            if target.imprisonment_bucket is not None
            and rows[i].get("task_masks", {}).get("sentencing_train", True)
        ]
        self.imprisonment_bucket_model.fit(
            matrix[month_indices], [targets[i].imprisonment_bucket for i in month_indices]
        )
        self.imprisonment_regressor.fit(
            matrix[month_indices],
            [math.log1p(targets[i].imprisonment_months or 0) for i in month_indices],
        )

        fine_known = [
            i
            for i, target in enumerate(targets)
            if target.fine_known and rows[i].get("task_masks", {}).get("fine_binary", True)
        ]
        self.fine_imposed_model.fit(
            matrix[fine_known], [int(bool(targets[i].fine_imposed)) for i in fine_known]
        )
        fine_positive = [
            i
            for i, target in enumerate(targets)
            if target.fine_bucket is not None
            and rows[i].get("task_masks", {}).get("fine_amount", True)
        ]
        self.fine_bucket_model.fit(
            matrix[fine_positive], [targets[i].fine_bucket for i in fine_positive]
        )
        self.fine_regressor.fit(
            matrix[fine_positive],
            [math.log1p(targets[i].fine_amount or 0) for i in fine_positive],
        )
        self.metadata = {
            "model_type": "hashing_char_ngram_hierarchical_sgd",
            "random_state": self.random_state,
            "training_rows": len(rows),
            "sentence_type_rows": len(type_indices),
            "imprisonment_rows": len(month_indices),
            "fine_known_rows": len(fine_known),
            "fine_positive_rows": len(fine_positive),
            "accusation_conditioning_dropout": 0.5,
            "limitations": [
                "legacy CAIL labels cannot distinguish fixed-term imprisonment, detention and control",
                "probation is not trained because no reliable probation label is available",
            ],
        }
        return self.metadata

    @staticmethod
    def _prediction(model, matrix) -> tuple[object, float]:
        probabilities = model.predict_proba(matrix)[0]
        index = int(np.argmax(probabilities))
        return model.classes_[index], float(probabilities[index])

    def predict_imprisonment_given_fixed(self, matrix) -> tuple[int, tuple[int, int]]:
        """Evaluate/use the duration head conditional on a fixed-term target."""
        month_bucket, _ = self._prediction(self.imprisonment_bucket_model, matrix)
        lower, upper = bucket_range(int(month_bucket), IMPRISONMENT_BINS)
        raw_months = max(1, round(math.expm1(self.imprisonment_regressor.predict(matrix)[0])))
        return min(max(raw_months, lower), upper), (lower, upper)

    def predict_fine_amount_given_imposed(self, matrix) -> tuple[int, tuple[int, int]]:
        """Evaluate/use the amount head conditional on a positive-fine target."""
        fine_bucket, _ = self._prediction(self.fine_bucket_model, matrix)
        lower, upper = bucket_range(int(fine_bucket), FINE_BINS)
        raw_fine = max(1, round(math.expm1(self.fine_regressor.predict(matrix)[0])))
        return min(max(raw_fine, lower), upper), (lower, upper)

    def predict(self, fact: str, accusations: list[str] | None = None) -> dict:
        matrix = self.vectorizer.transform([model_text(fact, accusations)])
        type_probabilities = self.sentence_type_model.predict_proba(matrix)[0]
        type_probability_map = {
            str(label): float(type_probabilities[index])
            for index, label in enumerate(self.sentence_type_model.classes_)
        }
        sentence_type = max(type_probability_map, key=type_probability_map.get)
        type_probability = type_probability_map[sentence_type]
        result = {
            "sentence_type": sentence_type,
            "sentence_type_probability": type_probability,
            "death_probability": type_probability_map.get("death", 0.0),
            "life_probability": type_probability_map.get("life", 0.0),
            "imprisonment_months": None,
            "imprisonment_range": None,
            "fine_imposed": None,
            "fine_probability": None,
            "fine_amount": None,
            "fine_range": None,
        }
        if sentence_type == "fixed_term":
            result["imprisonment_months"], result["imprisonment_range"] = (
                self.predict_imprisonment_given_fixed(matrix)
            )

        imposed_probabilities = self.fine_imposed_model.predict_proba(matrix)[0]
        positive_index = list(self.fine_imposed_model.classes_).index(1)
        fine_probability = float(imposed_probabilities[positive_index])
        result["fine_probability"] = fine_probability
        result["fine_imposed"] = fine_probability >= 0.5
        if result["fine_imposed"]:
            result["fine_amount"], result["fine_range"] = self.predict_fine_amount_given_imposed(
                matrix
            )
        return result

    def save(self, output_dir: str | Path) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target / "sentencing_baseline.joblib", compress=3)
        (target / "manifest.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, output_dir: str | Path) -> SentencingBaseline:
        return joblib.load(Path(output_dir) / "sentencing_baseline.joblib")
