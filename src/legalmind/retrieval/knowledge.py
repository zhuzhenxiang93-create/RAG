from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from legalmind.data.normalize import normalize_record
from legalmind.retrieval.chunker import ChineseCaseChunker
from legalmind.schemas import StatuteChunk

_ARTICLE_RE = re.compile(
    r"第(?P<number>[一二三四五六七八九十百千零〇0-9]+)条\s*(?P<text>.*?)(?=第[一二三四五六七八九十百千零〇0-9]+条|\Z)",
    re.DOTALL,
)


_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000}


def chinese_number(value: str) -> int:
    if value.isdigit():
        return int(value)
    total = current = 0
    for character in value:
        if character in _CHINESE_DIGITS:
            current = _CHINESE_DIGITS[character]
        elif character in _CHINESE_UNITS:
            unit = _CHINESE_UNITS[character]
            total += (current or 1) * unit
            current = 0
    return total + current


def parse_statute_text(
    text: str,
    source_url: str,
    version: str,
    source_status: str,
    verified_at: str | None = None,
) -> list[StatuteChunk]:
    rows: dict[int, StatuteChunk] = {}
    for match in _ARTICLE_RE.finditer(text):
        number = chinese_number(match.group("number"))
        body = re.sub(r"\s+", " ", match.group("text")).strip()
        if not body:
            continue
        candidate = StatuteChunk(
            statute_id=f"PRC-CRIMINAL-LAW-{number}",
            clause_id=f"PRC-CRIMINAL-LAW#{number}",
            law_name="中华人民共和国刑法",
            article_number=number,
            title=f"第{number}条",
            text=body,
            source_url=source_url,
            version=version,
            verified_at=verified_at,
            source_status=source_status,
        )
        current = rows.get(number)
        if current is None or len(candidate.text) > len(current.text):
            rows[number] = candidate
    return [rows[number] for number in sorted(rows)]


def build_case_knowledge(
    source: Path,
    documents_path: Path,
    chunks_path: Path,
    tokenizer,
    chunk_size: int = 1024,
    overlap: int = 128,
    max_cases: int | None = None,
) -> dict:
    chunker = ChineseCaseChunker(chunk_size, overlap, tokenizer=tokenizer)
    documents_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    cases = chunks = 0
    source_digest = hashlib.sha256()
    with (
        source.open("rb") as raw,
        documents_path.open("w", encoding="utf-8", newline="\n") as docs,
        chunks_path.open("w", encoding="utf-8", newline="\n") as chunk_file,
    ):
        for line in raw:
            if not line.strip():
                continue
            if max_cases and cases >= max_cases:
                break
            source_digest.update(line)
            row = json.loads(line)
            case = normalize_record(row, source_split="train")
            docs.write(json.dumps(case.model_dump(), ensure_ascii=False) + "\n")
            for chunk in chunker.split(case):
                chunk_file.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
                chunks += 1
            cases += 1
    return {
        "source": str(source),
        "source_prefix_sha256": source_digest.hexdigest(),
        "cases": cases,
        "chunks": chunks,
        "chunk_size_tokens": chunk_size,
        "chunk_overlap_tokens": overlap,
        "split": "train_only",
    }
