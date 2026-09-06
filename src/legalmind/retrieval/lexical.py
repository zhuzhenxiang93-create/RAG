from __future__ import annotations

import heapq
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from legalmind.schemas import CaseChunk, SearchHit


def chinese_bigrams(text: str) -> list[str]:
    compact = "".join(text.split())
    if len(compact) < 2:
        return [compact] if compact else []
    return [compact[index : index + 2] for index in range(len(compact) - 1)]


class LexicalBM25Index:
    """Sparse inverted BM25 that scores only documents containing query terms."""

    def __init__(self) -> None:
        self.chunks: list[CaseChunk] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}
        self.doc_lengths: list[int] = []
        self.average_doc_length = 0.0
        self.k1 = 1.5
        self.b = 0.75

    def build(self, chunks: list[CaseChunk]) -> None:
        self.chunks = chunks
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        self.doc_lengths = []
        for document_id, chunk in enumerate(chunks):
            frequencies = Counter(chinese_bigrams(chunk.text))
            self.doc_lengths.append(sum(frequencies.values()))
            document_frequency.update(frequencies)
            for token, frequency in frequencies.items():
                postings[token].append((document_id, frequency))
        self.postings = dict(postings)
        document_count = len(chunks)
        self.average_doc_length = sum(self.doc_lengths) / max(document_count, 1)
        self.idf = {
            token: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 20) -> list[SearchHit]:
        if not self.postings:
            raise RuntimeError("BM25 index is not built")
        scores: dict[int, float] = defaultdict(float)
        query_terms = Counter(chinese_bigrams(query))
        informative_terms = sorted(
            (
                (token, frequency, self.idf[token])
                for token, frequency in query_terms.items()
                if self.idf.get(token, 0.0) >= 0.2
            ),
            key=lambda item: item[2] * min(item[1], 3),
            reverse=True,
        )[:64]
        # In very small charge partitions, every useful token can have IDF below
        # the global-noise cutoff. Keep BM25 semantics by falling back to all
        # indexed query terms instead of returning an empty candidate set.
        if not informative_terms:
            informative_terms = sorted(
                (
                    (token, frequency, self.idf[token])
                    for token, frequency in query_terms.items()
                    if token in self.idf
                ),
                key=lambda item: item[2] * min(item[1], 3),
                reverse=True,
            )[:64]
        for token, query_frequency, idf in informative_terms:
            for document_id, term_frequency in self.postings[token]:
                length_normalization = self.k1 * (
                    1.0
                    - self.b
                    + self.b * self.doc_lengths[document_id] / max(self.average_doc_length, 1.0)
                )
                scores[document_id] += (
                    idf
                    * term_frequency
                    * (self.k1 + 1.0)
                    / (term_frequency + length_normalization)
                    * min(query_frequency, 3)
                )
        ranked = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
        return [
            SearchHit(**self.chunks[document_id].model_dump(), score=float(score))
            for document_id, score in ranked
        ]

    def save(self, output_dir: str | Path) -> None:
        if not self.postings:
            raise RuntimeError("BM25 index is not built")
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
        with (target / "bm25.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "postings": self.postings,
                    "idf": self.idf,
                    "doc_lengths": self.doc_lengths,
                    "average_doc_length": self.average_doc_length,
                    "k1": self.k1,
                    "b": self.b,
                },
                handle,
            )
        (target / "manifest.json").write_text(
            json.dumps(
                {
                    "index_type": "sparse_inverted_BM25",
                    "tokenization": "Chinese character bigram",
                    "chunks": len(self.chunks),
                    "vocabulary": len(self.postings),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: str | Path) -> LexicalBM25Index:
        source = Path(output_dir)
        instance = cls()
        with (source / "chunks.jsonl").open(encoding="utf-8") as handle:
            instance.chunks = [
                CaseChunk.model_validate_json(line) for line in handle if line.strip()
            ]
        with (source / "bm25.pkl").open("rb") as handle:
            state = pickle.load(handle)
        instance.postings = state["postings"]
        instance.idf = state["idf"]
        instance.doc_lengths = state["doc_lengths"]
        instance.average_doc_length = state["average_doc_length"]
        instance.k1 = state["k1"]
        instance.b = state["b"]
        return instance
