"""Export EnterpriseRAG-Bench JSON documents into retrieval-ready text files."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Set


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gold-paths",
        type=Path,
        help="Optional rg-generated file list containing every gold document path.",
    )
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="Export only documents referenced by the benchmark gold labels.",
    )
    parser.add_argument(
        "--distractors-per-source",
        type=int,
        default=5000,
        help="Non-gold documents retained per source; all gold documents are retained.",
    )
    args = parser.parse_args()

    questions = list(read_jsonl(args.questions))
    args.output.mkdir(parents=True, exist_ok=True)
    packed_path = args.output / "corpus.jsonl"
    packed_handle = packed_path.open("w", encoding="utf-8", newline="\n")
    expected_ids: Set[str] = {
        str(document_id)
        for row in questions
        for document_id in row.get("expected_doc_ids", [])
    }
    found_ids: Set[str] = set()
    gold_paths = set()
    if args.gold_paths:
        gold_paths = {
            os.path.normcase(os.path.abspath(line.strip()))
            for line in args.gold_paths.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    counts: Dict[str, int] = {}
    scanned = 0
    parse_errors = 0
    for source_dir in sorted(path for path in args.sources_dir.iterdir() if path.is_dir()):
        source = source_dir.name.replace("-", "_")
        exported = 0
        distractors = 0
        written_ids: Set[str] = set()
        paths = sorted(source_dir.rglob("*.json"))
        if gold_paths:
            distractor_paths = [] if args.gold_only else paths[: args.distractors_per_source]
            source_gold_paths = [
                path for path in paths if os.path.normcase(str(path)) in gold_paths
            ]
            paths = sorted(set(distractor_paths) | set(source_gold_paths))
        for path in paths:
            scanned += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                parse_errors += 1
                continue
            document_id = str(payload.get("dataset_doc_uuid") or "")
            if not document_id:
                continue
            if document_id in written_ids:
                continue
            if document_id in expected_ids:
                found_ids.add(document_id)
            if args.gold_only and document_id not in expected_ids:
                continue
            if document_id not in expected_ids:
                if distractors >= args.distractors_per_source:
                    continue
                distractors += 1
            title = str(payload.get("title") or payload.get("name") or path.stem)
            content = str(payload.get("content") or "")
            if not content:
                fields = payload.get("content_field_names") or []
                content = "\n\n".join(str(payload.get(field) or "") for field in fields)
            destination = args.output / source / f"{document_id}.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            rendered = f"# {title}\n\n{content}".strip() + "\n"
            destination.write_text(rendered, encoding="utf-8")
            packed_handle.write(
                json.dumps(
                    {
                        "document_id": document_id,
                        "source": source,
                        "filename": destination.name,
                        "content": rendered,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written_ids.add(document_id)
            exported += 1
        counts[source] = exported

    packed_handle.close()
    missing = sorted(expected_ids - found_ids)
    manifest = {
        "source": "EnterpriseRAG-Bench generated_data/sources",
        "gold_only": args.gold_only,
        "distractors_per_source": 0 if args.gold_only else args.distractors_per_source,
        "gold_path_prefilter": bool(args.gold_paths),
        "scanned_json_documents": scanned,
        "exported_documents": sum(counts.values()),
        "exported_by_source": counts,
        "packed_corpus": str(packed_path),
        "parse_errors": parse_errors,
        "expected_gold_document_count": len(expected_ids),
        "found_gold_document_count": len(found_ids),
        "gold_document_coverage": (
            round(len(found_ids) / len(expected_ids), 6) if expected_ids else 1.0
        ),
        "missing_gold_document_ids": missing,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**manifest, "missing_gold_document_ids": missing[:20]}, indent=2))
    if missing:
        raise SystemExit(f"{len(missing)} benchmark gold documents were not found")


if __name__ == "__main__":
    main()
