from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalmind.retrieval.lexical import LexicalBM25Index
from legalmind.schemas import CaseChunk, StatuteChunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statutes", default="data/knowledge/statutes/statute_chunks.jsonl")
    parser.add_argument("--output", default="artifacts/indexes/statutes_bm25_v2")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite statute index: {output}")
    chunks: list[CaseChunk] = []
    with Path(args.statutes).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            statute = StatuteChunk.model_validate_json(line)
            text = f"{statute.law_name}第{statute.article_number}条 {statute.title} {statute.text}"
            chunks.append(
                CaseChunk(
                    chunk_id=statute.clause_id,
                    case_id=statute.statute_id,
                    text=text,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(text),
                    accusations=[],
                    relevant_articles=[statute.article_number],
                    source_split="public_statute_unreviewed",
                    parent_title=statute.title,
                    evidence_type="statute",
                    source_url=statute.source_url,
                    promulgation_date=statute.promulgation_date,
                    effective_date=statute.effective_from,
                    expiry_date=statute.expiry_date,
                    legal_status=statute.status,
                    source_status=statute.source_status,
                    retrieved_at=statute.retrieved_at,
                    checksum=statute.checksum,
                )
            )
    index = LexicalBM25Index()
    index.build(chunks)
    index.save(output)
    manifest_path = output / "statute_source_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "statutes": len(chunks),
                "source": args.statutes,
                "source_review_status": "official_government_url_automatically_downloaded_unreviewed",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
