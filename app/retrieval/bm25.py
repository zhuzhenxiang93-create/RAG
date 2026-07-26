"""Small dependency-free BM25 implementation for Lite mode."""

import math
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, List, Tuple

from app.retrieval.models import IndexedChunk
from app.retrieval.tokenizer import tokenize


class BM25Index:
    """In-memory Okapi BM25 index with an inverted posting list."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: List[IndexedChunk] = []
        self.term_frequencies: List[Counter] = []
        self.document_frequency: Counter = Counter()
        self.postings: DefaultDict[str, List[int]] = defaultdict(list)
        self.lengths: List[int] = []
        self.average_length = 0.0

    def build(self, chunks: List[IndexedChunk]) -> None:
        self.chunks = list(chunks)
        self.term_frequencies = []
        self.document_frequency = Counter()
        self.postings = defaultdict(list)
        self.lengths = []

        for index, chunk in enumerate(self.chunks):
            frequencies = Counter(tokenize(chunk.content))
            self.term_frequencies.append(frequencies)
            self.lengths.append(sum(frequencies.values()))
            for term in frequencies:
                self.document_frequency[term] += 1
                self.postings[term].append(index)
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )

    def search(self, query: str, limit: int) -> List[Tuple[IndexedChunk, float]]:
        if not self.chunks:
            return []
        scores: Dict[int, float] = defaultdict(float)
        query_terms = Counter(tokenize(query))
        total = len(self.chunks)
        average = self.average_length or 1.0

        for term, query_frequency in query_terms.items():
            frequency = self.document_frequency.get(term, 0)
            if frequency == 0:
                continue
            inverse_document_frequency = math.log(
                1.0 + (total - frequency + 0.5) / (frequency + 0.5)
            )
            for index in self.postings[term]:
                term_frequency = self.term_frequencies[index][term]
                length_ratio = self.lengths[index] / average
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length_ratio
                )
                scores[index] += (
                    inverse_document_frequency
                    * (term_frequency * (self.k1 + 1.0) / denominator)
                    * query_frequency
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [(self.chunks[index], score) for index, score in ranked]
