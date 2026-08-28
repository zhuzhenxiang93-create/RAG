from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def label_support(records: list[dict[str, Any]], num_labels: int) -> np.ndarray:
    support = np.zeros(num_labels, dtype=np.int64)
    for record in records:
        for label_id in set(record.get("accusation_ids", [])):
            label_id = int(label_id)
            if 0 <= label_id < num_labels:
                support[label_id] += 1
    return support


def subset_report(
    records: list[dict[str, Any]],
    num_labels: int,
    *,
    source_rows: int | None = None,
    selected_indices: list[int] | None = None,
) -> dict[str, Any]:
    support = label_support(records, num_labels)
    present = np.flatnonzero(support > 0)
    report: dict[str, Any] = {
        "rows": len(records),
        "source_rows": source_rows if source_rows is not None else len(records),
        "num_labels": num_labels,
        "labels_present": int(present.size),
        "labels_missing": np.flatnonzero(support == 0).tolist(),
        "label_support": support.tolist(),
        "multi_label_rows": sum(
            len(set(record.get("accusation_ids", []))) > 1 for record in records
        ),
    }
    if selected_indices is not None:
        payload = ",".join(str(index) for index in selected_indices).encode()
        report["selected_indices_sha256"] = hashlib.sha256(payload).hexdigest()
    return report


def select_multilabel_subset(
    records: list[dict[str, Any]],
    sample_size: int,
    num_labels: int,
    *,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic, coverage-first iterative multilabel subset."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size >= len(records):
        indices = list(range(len(records)))
        return list(records), subset_report(
            records,
            num_labels,
            source_rows=len(records),
            selected_indices=indices,
        )

    labels_by_index = [
        {
            int(label_id)
            for label_id in record.get("accusation_ids", [])
            if 0 <= int(label_id) < num_labels
        }
        for record in records
    ]
    full_support = label_support(records, num_labels)
    indices_by_label: list[list[int]] = [[] for _ in range(num_labels)]
    for index, labels in enumerate(labels_by_index):
        for label_id in labels:
            indices_by_label[label_id].append(index)

    rng = np.random.default_rng(seed)
    tie_break = rng.permutation(len(records))
    tie_rank = np.empty(len(records), dtype=np.int64)
    tie_rank[tie_break] = np.arange(len(records))
    selected: set[int] = set()
    covered: set[int] = set()

    # Seed the subset with rare labels first. A multi-label row may cover several
    # labels, so prefer the row that adds the most currently uncovered labels.
    for label_id in sorted(range(num_labels), key=lambda value: (full_support[value], value)):
        if full_support[label_id] == 0 or label_id in covered or len(selected) >= sample_size:
            continue
        candidates = [index for index in indices_by_label[label_id] if index not in selected]
        if not candidates:
            continue
        chosen = min(
            candidates,
            key=lambda index: (
                -len(labels_by_index[index] - covered),
                sum(full_support[value] for value in labels_by_index[index]),
                int(tie_rank[index]),
            ),
        )
        selected.add(chosen)
        covered.update(labels_by_index[chosen])

    remaining_slots = sample_size - len(selected)
    if remaining_slots:
        candidate_indices = [index for index in range(len(records)) if index not in selected]
        if remaining_slots >= len(candidate_indices):
            selected.update(candidate_indices)
        else:
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

            labels = np.zeros((len(candidate_indices), num_labels), dtype=np.int8)
            for row, index in enumerate(candidate_indices):
                labels[row, list(labels_by_index[index])] = 1
            splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                train_size=remaining_slots,
                test_size=len(candidate_indices) - remaining_slots,
                random_state=seed,
            )
            chosen_local, _ = next(splitter.split(np.zeros(len(candidate_indices)), labels))
            chosen_global = [candidate_indices[int(index)] for index in chosen_local]
            if len(chosen_global) > remaining_slots:
                chosen_global = sorted(
                    chosen_global, key=lambda index: int(tie_rank[index])
                )[:remaining_slots]
            elif len(chosen_global) < remaining_slots:
                chosen_set = set(chosen_global)
                fill = sorted(
                    (index for index in candidate_indices if index not in chosen_set),
                    key=lambda index: int(tie_rank[index]),
                )
                chosen_global.extend(fill[: remaining_slots - len(chosen_global)])
            selected.update(chosen_global)

    selected_indices = sorted(selected)
    sampled = [records[index] for index in selected_indices]
    report = subset_report(
        sampled,
        num_labels,
        source_rows=len(records),
        selected_indices=selected_indices,
    )
    report.update(
        {
            "strategy": "rare-label coverage + iterative multilabel stratification",
            "seed": seed,
            "source_label_support": full_support.tolist(),
        }
    )
    return sampled, report
