from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalmind.retrieval.knowledge import build_case_knowledge, parse_statute_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-source", default="data/processed_v2/train.jsonl")
    parser.add_argument("--output-dir", default="data/knowledge")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--skip-cases", action="store_true")
    parser.add_argument("--statute-text")
    parser.add_argument(
        "--statute-source-url",
        default="https://flk.npc.gov.cn/detail?fileId=&id=ff808181796a636a0179822a19640c92&type=",
    )
    parser.add_argument(
        "--statute-version", default="2024-03-01-consolidation-pending-verification"
    )
    parser.add_argument(
        "--statute-source-status",
        default="official_government_url_automatically_downloaded_unreviewed",
    )
    args = parser.parse_args()
    from transformers import AutoTokenizer

    output = Path(args.output_dir)
    if args.skip_cases:
        existing = output / "manifests/knowledge_manifest.json"
        manifest = json.loads(existing.read_text(encoding="utf-8")) if existing.exists() else {}
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        manifest = build_case_knowledge(
            Path(args.case_source),
            output / "cases/case_documents.jsonl",
            output / "cases/case_chunks.jsonl",
            tokenizer,
            max_cases=args.max_cases,
        )
    statute_status = "not_built"
    if args.statute_text:
        statute_source = Path(args.statute_text)
        rows = parse_statute_text(
            statute_source.read_text(encoding="utf-8"),
            args.statute_source_url,
            args.statute_version,
            args.statute_source_status,
        )
        target = output / "statutes/statute_chunks.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")
        statute_status = {"rows": len(rows), "source": str(statute_source)}
    manifest["statutes"] = statute_status
    target = output / "manifests/knowledge_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
