from __future__ import annotations

import os
from typing import Any, Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    def encode_documents(self, texts: list[str], batch_size: int) -> np.ndarray: ...

    def encode_query(self, query: str) -> np.ndarray: ...


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


class LocalEmbeddingBackend:
    def __init__(
        self,
        model_name: str,
        query_instruction: str = "",
        max_length: int = 2048,
        dimensions: int | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_length
        self.query_instruction = query_instruction.strip()
        self.dimensions = dimensions

    def encode_documents(self, texts: list[str], batch_size: int) -> np.ndarray:
        values = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            truncate_dim=self.dimensions,
        )
        return np.asarray(values, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        query_text = (
            f"Instruct: {self.query_instruction}\nQuery: {query}"
            if self.query_instruction
            else query
        )
        values = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            truncate_dim=self.dimensions,
        )
        return np.asarray(values, dtype=np.float32)


class OpenAIEmbeddingBackend:
    """OpenAI-compatible embedding backend used by Alibaba Model Studio."""

    def __init__(
        self,
        model_name: str,
        base_url_env: str,
        api_key_env: str,
        dimensions: int | None = 1024,
        query_instruction: str = "",
        api_batch_size: int = 20,
        client: Any | None = None,
    ):
        base_url = os.getenv(base_url_env)
        api_key = os.getenv(api_key_env)
        if client is None:
            if not base_url:
                raise ValueError(
                    f"Embedding base URL is missing from environment variable {base_url_env}"
                )
            if not api_key:
                raise ValueError(
                    f"Embedding API key is missing from environment variable {api_key_env}"
                )
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = client
        self.model_name = model_name
        self.dimensions = dimensions
        self.query_instruction = query_instruction.strip()
        self.api_batch_size = api_batch_size

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        rows: list[list[float]] = []
        effective_batch = max(1, min(batch_size, self.api_batch_size))
        for start in range(0, len(texts), effective_batch):
            request: dict[str, Any] = {
                "model": self.model_name,
                "input": texts[start : start + effective_batch],
                "encoding_format": "float",
            }
            if self.dimensions is not None:
                request["dimensions"] = self.dimensions
            response = self.client.embeddings.create(**request)
            batch = sorted(response.data, key=lambda item: item.index)
            rows.extend(item.embedding for item in batch)
        if len(rows) != len(texts):
            raise ValueError("Embedding service returned a vector count that does not match inputs")
        return _normalize(np.asarray(rows, dtype=np.float32))

    def encode_documents(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._encode(texts, batch_size)

    def encode_query(self, query: str) -> np.ndarray:
        query_text = (
            f"Instruct: {self.query_instruction}\nQuery: {query}"
            if self.query_instruction
            else query
        )
        return self._encode([query_text], 1)
