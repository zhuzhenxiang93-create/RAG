"""Deterministic hashing-based dense baseline for Lite mode."""

import hashlib
from typing import List, Tuple

import numpy as np

from app.retrieval.models import IndexedChunk
from app.retrieval.tokenizer import tokenize


class LiteDenseIndex:
    """Feature-hashing vectors used as a reproducible CPU baseline."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.chunks: List[IndexedChunk] = []
        self.matrix = np.empty((0, dimensions), dtype=np.float32)

    def build(self, chunks: List[IndexedChunk]) -> None:
        self.chunks = list(chunks)
        if not chunks:
            self.matrix = np.empty((0, self.dimensions), dtype=np.float32)
            return
        self.matrix = np.vstack([self.embed(chunk.content) for chunk in chunks])

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = tokenize(text)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            sign = 1.0 if (value >> 8) & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def search(self, query: str, limit: int) -> List[Tuple[IndexedChunk, float]]:
        if not self.chunks:
            return []
        query_vector = self.embed(query)
        if not np.any(query_vector):
            return []
        scores = self.matrix @ query_vector
        indices = np.argsort(-scores, kind="stable")[:limit]
        return [
            (self.chunks[int(index)], float(scores[int(index)]))
            for index in indices
            if scores[int(index)] > 0
        ]
