from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse

from legalmind.data.money import parse_rmb
from legalmind.data.privacy import redact_privacy
from legalmind.sentencing.labels import labels_from_row, sentence_type_from_labels
from legalmind.sentencing.model import SentencingBaseline, model_text
from legalmind.sentencing.schemas import SimilarCase

_AMOUNT = re.compile(
    r"(?:价值|金额|价款|损失|共计|合计|人民币)\s*"
    r"([0-9]+(?:\.[0-9]+)?|[零〇一二两三四五六七八九十百千万]+)\s*(万元|万|元)"
)
_FACTORS = ("自首", "坦白", "退赃", "退赔", "谅解", "累犯", "前科", "未遂", "认罪认罚")


def _features(text: str) -> tuple[int | None, set[str]]:
    match = _AMOUNT.search(text)
    amount = parse_rmb("".join(match.groups())).amount if match else None
    return amount, {factor for factor in _FACTORS if factor in text}


def _structured_score(
    query_amount: int | None,
    query_factors: set[str],
    candidate: dict,
) -> float:
    candidate_amount = candidate.get("amount_feature")
    if query_amount is None or candidate_amount is None:
        amount_score = 0.5
    else:
        amount_score = float(np.exp(-abs(np.log1p(query_amount) - np.log1p(candidate_amount))))
    candidate_factors = set(candidate.get("factor_features", []))
    union = query_factors | candidate_factors
    factor_score = len(query_factors & candidate_factors) / len(union) if union else 0.5
    return 0.6 * amount_score + 0.4 * factor_score


def _summary(text: str, limit: int = 120) -> str:
    compact = " ".join(redact_privacy(text).text.split())
    # Avoid exposing names while retaining the sentencing facts useful for explanation.
    compact = re.sub(
        r"(被告人|被害人)([\u4e00-\u9fff]{1,4})(?=于|在|，|。|、|系|窜|伙同|到|$)",
        r"\1某",
        compact,
    )
    compact = compact.replace("被告人", "当事人").replace("被害人", "相关人员")
    return compact[:limit] + ("…" if len(compact) > limit else "")


class SentencingEvidenceIndex:
    def __init__(
        self,
        model: SentencingBaseline,
        matrix=None,
        metadata: list[dict] | None = None,
        min_score: float = 0.20,
    ):
        self.model = model
        self.matrix = matrix
        self.metadata = metadata or []
        self.min_score = min_score

    def build(self, rows: list[dict]) -> None:
        self.matrix = self.model.vectorizer.transform(
            [model_text(row["fact"], labels_from_row(row).get("accusations", [])) for row in rows]
        ).tocsr()
        self.metadata = []
        for row in rows:
            labels = labels_from_row(row)
            fine = labels.get("fine")
            if isinstance(fine, dict):
                fine = fine.get("amount") if fine.get("status") in {"zero", "positive"} else None
            amount_feature, factor_features = _features(row["fact"])
            self.metadata.append(
                {
                    "case_id": row["case_id"],
                    "accusations": labels.get("accusations", []),
                    "sentence_type": sentence_type_from_labels(labels).value,
                    "imprisonment_months": labels.get("imprisonment_months"),
                    "fine": fine,
                    "summary": _summary(row["fact"]),
                    "amount_feature": amount_feature,
                    "factor_features": sorted(factor_features),
                }
            )

    def search(self, fact: str, accusations: list[str], top_k: int = 3) -> list[SimilarCase]:
        if self.matrix is None or not self.metadata or top_k <= 0:
            return []
        query = self.model.vectorizer.transform([model_text(fact, accusations)])
        scores = (self.matrix @ query.T).toarray().ravel()
        requested = set(accusations)
        eligible = np.array(
            [
                index
                for index, item in enumerate(self.metadata)
                if not requested or requested.intersection(item["accusations"])
            ],
            dtype=np.int64,
        )
        if not len(eligible):
            return []
        count = min(max(top_k * 20, top_k), len(eligible))
        eligible_scores = scores[eligible]
        candidate_positions = np.argpartition(eligible_scores, -count)[-count:]
        candidates = eligible[candidate_positions]
        query_amount, query_factors = _features(fact)
        ranked = []
        for index in candidates:
            item = self.metadata[int(index)]
            vector_score = max(0.0, min(1.0, float(scores[index])))
            structure = _structured_score(query_amount, query_factors, item)
            exact_charge = float(set(item["accusations"]) == requested)
            score = 0.70 * vector_score + 0.20 * structure + 0.10 * exact_charge
            ranked.append((score, int(index)))
        ranked.sort(reverse=True)
        results = []
        seen_case_ids: set[str] = set()
        for score, index in ranked:
            item = self.metadata[index]
            if score < self.min_score or item["case_id"] in seen_case_ids:
                continue
            seen_case_ids.add(item["case_id"])
            payload = {
                key: value
                for key, value in item.items()
                if key not in {"amount_feature", "factor_features"}
            }
            results.append(SimilarCase(**payload, similarity_score=min(1.0, score)))
            if len(results) == min(top_k, 3):
                break
        return results

    def save(self, output_dir: str | Path) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(target / "evidence_matrix.npz", self.matrix)
        with (target / "evidence_metadata.jsonl").open("w", encoding="utf-8") as handle:
            for item in self.metadata:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, output_dir: str | Path, model: SentencingBaseline) -> SentencingEvidenceIndex:
        source = Path(output_dir)
        matrix = sparse.load_npz(source / "evidence_matrix.npz")
        with (source / "evidence_metadata.jsonl").open(encoding="utf-8") as handle:
            metadata = [json.loads(line) for line in handle if line.strip()]
        return cls(model=model, matrix=matrix, metadata=metadata)
