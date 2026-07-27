"""Dataset normalization and contamination checks for router training."""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple

from app.plugins.enterprise_router.schema import (
    QUESTION_TYPES,
    SOURCE_TYPES,
    RouteDecision,
)


ROUTER_SYSTEM_PROMPT = (
    "You are a retrieval router for a multi-source enterprise knowledge base. "
    "Return exactly one compact JSON object matching the requested schema. "
    "Do not answer the user's question. "
    "Type guide: basic=one direct fact; semantic=conceptual or paraphrased lookup; "
    "intra_document_reasoning=combine facts inside one document; "
    "project_related=synthesize a project across documents; constrained=apply explicit "
    "filters; conflicting_info=resolve inconsistent claims; completeness=find every "
    "relevant item; miscellaneous=general orientation; high_level=organization-wide "
    "summary; info_not_found=the requested fact is absent. "
    "Source cues: chat/channel=slack, email/thread=gmail, issue/project=linear, shared "
    "document/sheet=google_drive, customer/deal=hubspot, meeting/call=fireflies, "
    "repository/commit/PR=github, sprint/ticket=Jira, wiki/policy/runbook=confluence. "
    "Use every source when the question itself does not support safe source pruning."
)

_TYPE_ALIASES = {
    "intra-document reasoning": "intra_document_reasoning",
    "intra_document_reasoning": "intra_document_reasoning",
    "project related": "project_related",
    "project_related": "project_related",
    "conflicting info": "conflicting_info",
    "conflicting_info": "conflicting_info",
    "info not found": "info_not_found",
    "info_not_found": "info_not_found",
    "high level": "high_level",
    "high_level": "high_level",
}


def normalize_label(value: str) -> str:
    normalized = re.sub(r"[-\s]+", "_", str(value).strip().lower())
    return _TYPE_ALIASES.get(str(value).strip().lower(), normalized)


def route_from_question(row: Dict) -> RouteDecision:
    """Derive supervised routing labels from an EnterpriseRAG-style question."""
    question_type = normalize_label(row["question_type"])
    if question_type not in QUESTION_TYPES:
        raise ValueError("Unsupported question_type: {}".format(question_type))
    sources = [normalize_label(item) for item in row.get("source_types", [])]
    invalid_sources = sorted(set(sources) - set(SOURCE_TYPES))
    if invalid_sources:
        raise ValueError("Unsupported source type(s): {}".format(", ".join(invalid_sources)))
    if not sources:
        # High-level and not-found questions may not have a gold source. The broad
        # route is explicit rather than silently inventing a single source.
        sources = list(SOURCE_TYPES)

    expected_docs = row.get("expected_doc_ids") or []
    multi_document = len(expected_docs) > 1 or question_type in {
        "project_related",
        "conflicting_info",
        "completeness",
        "high_level",
    }
    conflict_check = question_type == "conflicting_info"
    completeness_required = question_type in {
        "project_related",
        "conflicting_info",
        "completeness",
        "high_level",
    }
    if question_type in {"basic", "semantic", "miscellaneous"}:
        retrieval_depth = "shallow"
    elif question_type in {
        "project_related",
        "conflicting_info",
        "completeness",
        "high_level",
    }:
        retrieval_depth = "deep"
    else:
        retrieval_depth = "standard"
    answerability = (
        "unanswerable"
        if question_type == "info_not_found"
        else "unknown"
        if question_type == "high_level"
        else "answerable"
    )
    return RouteDecision(
        question_type=question_type,
        sources=sources,
        multi_document=multi_document,
        conflict_check=conflict_check,
        completeness_required=completeness_required,
        retrieval_depth=retrieval_depth,
        answerability=answerability,
    )


def normalized_question_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def question_fingerprint(text: str) -> str:
    return hashlib.sha256(normalized_question_text(text).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("{}:{} is invalid JSON".format(path, line_number)) from exc


def benchmark_signatures(rows: Iterable[Dict]) -> Tuple[Set[str], Set[str]]:
    identifiers: Set[str] = set()
    fingerprints: Set[str] = set()
    for row in rows:
        if row.get("question_id"):
            identifiers.add(str(row["question_id"]))
        fingerprints.add(question_fingerprint(row["question"]))
    return identifiers, fingerprints


def assert_no_benchmark_leakage(
    training_rows: Sequence[Dict], benchmark_rows: Sequence[Dict]
) -> None:
    benchmark_ids, benchmark_hashes = benchmark_signatures(benchmark_rows)
    collisions = []
    for index, row in enumerate(training_rows):
        identifier = str(row.get("question_id", ""))
        fingerprint = question_fingerprint(row["question"])
        if identifier and identifier in benchmark_ids:
            collisions.append("row {} repeats question_id {}".format(index, identifier))
        if fingerprint in benchmark_hashes:
            collisions.append("row {} repeats benchmark question text".format(index))
    if collisions:
        raise ValueError(
            "Benchmark contamination detected; official questions must remain evaluation-only: "
            + "; ".join(collisions[:10])
        )


def format_instruction(question: str) -> str:
    schema = (
        '{"question_type":"...","sources":["..."],"multi_document":false,'
        '"conflict_check":false,"completeness_required":false,'
        '"retrieval_depth":"shallow|standard|deep",'
        '"answerability":"answerable|unknown|unanswerable"}'
    )
    return "{}\nSchema: {}\nQuestion: {}\nRoute:".format(
        ROUTER_SYSTEM_PROMPT, schema, question.strip()
    )


def normalized_training_row(row: Dict) -> Dict:
    decision = route_from_question(row)
    return {
        "id": str(row.get("question_id") or row.get("id") or ""),
        "question": row["question"].strip(),
        "prompt": format_instruction(row["question"]),
        "target": decision.canonical_json(),
        "route": decision.model_dump(),
        "question_fingerprint": question_fingerprint(row["question"]),
    }


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
