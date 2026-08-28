from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from legalmind.retrieval.embedding import (
    EmbeddingBackend,
    LocalEmbeddingBackend,
    OpenAIEmbeddingBackend,
)
from legalmind.schemas import CaseChunk, SearchHit


class HybridIndex:
    def __init__(
        self,
        embedding_model: str,
        embedding_provider: str = "local",
        query_instruction: str = "",
        max_length: int = 2048,
        embedding_dimension: int | None = None,
        base_url_env: str = "DASHSCOPE_BASE_URL",
        api_key_env: str = "DASHSCOPE_API_KEY",
        api_batch_size: int = 20,
        backend: EmbeddingBackend | None = None,
    ):
        import faiss
        from rank_bm25 import BM25Okapi

        self.faiss = faiss
        self.BM25Okapi = BM25Okapi
        if backend is not None:
            self.encoder = backend
        elif embedding_provider == "api":
            self.encoder = OpenAIEmbeddingBackend(
                embedding_model,
                base_url_env,
                api_key_env,
                dimensions=embedding_dimension,
                query_instruction=query_instruction,
                api_batch_size=api_batch_size,
            )
        elif embedding_provider == "local":
            self.encoder = LocalEmbeddingBackend(
                embedding_model,
                query_instruction=query_instruction,
                max_length=max_length,
                dimensions=embedding_dimension,
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.query_instruction = query_instruction.strip()
        self.max_length = max_length
        self.embedding_dimension = embedding_dimension
        self.base_url_env = base_url_env
        self.api_key_env = api_key_env
        self.api_batch_size = api_batch_size
        self.chunks: list[CaseChunk] = []
        self.bm25 = None
        self.vector_index = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        # Character bigrams work reliably for Chinese without a separate segmenter.
        compact = "".join(text.split())
        return [compact[index : index + 2] for index in range(max(1, len(compact) - 1))]

    def build(self, chunks: list[CaseChunk], batch_size: int = 64) -> None:
        self.chunks = chunks
        self.bm25 = self.BM25Okapi([self.tokenize(chunk.text) for chunk in chunks])
        values = self.encoder.encode_documents([chunk.text for chunk in chunks], batch_size)
        embeddings = np.ascontiguousarray(values, dtype=np.float32)
        self.vector_index = self.faiss.IndexFlatIP(embeddings.shape[1])
        self.vector_index.add(embeddings)

    def search_bm25(self, query: str, top_k: int) -> list[SearchHit]:
        if self.bm25 is None:
            raise RuntimeError("Index has not been built")
        scores = np.asarray(self.bm25.get_scores(self.tokenize(query)))
        indices = np.argsort(-scores)[:top_k]
        return [self._hit(int(index), float(scores[index])) for index in indices]

    def search_vector(self, query: str, top_k: int) -> list[SearchHit]:
        if self.vector_index is None:
            raise RuntimeError("Index has not been built")
        query_vector = self.encoder.encode_query(query)
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        scores, indices = self.vector_index.search(query_vector, top_k)
        return [
            self._hit(int(index), float(score))
            for index, score in zip(indices[0], scores[0])
            if index >= 0
        ]

    def _hit(self, index: int, score: float) -> SearchHit:
        chunk = self.chunks[index]
        return SearchHit(**chunk.model_dump(), score=score)

    def save(self, output_dir: str | Path) -> None:
        if self.bm25 is None or self.vector_index is None:
            raise RuntimeError("Index has not been built")
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
        self.faiss.write_index(self.vector_index, str(target / "vectors.faiss"))
        with (target / "bm25.pkl").open("wb") as handle:
            pickle.dump(self.bm25, handle)
        (target / "manifest.json").write_text(
            json.dumps(
                {
                    "embedding_model": self.embedding_model,
                    "embedding_provider": self.embedding_provider,
                    "query_instruction": self.query_instruction,
                    "max_length": self.max_length,
                    "embedding_dimension": self.embedding_dimension,
                    "base_url_env": self.base_url_env,
                    "api_key_env": self.api_key_env,
                    "api_batch_size": self.api_batch_size,
                    "chunks": len(self.chunks),
                    "vector_dimension": self.vector_index.d,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: str | Path) -> HybridIndex:
        source = Path(output_dir)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        instance = cls(
            manifest["embedding_model"],
            embedding_provider=manifest.get("embedding_provider", "local"),
            query_instruction=manifest.get("query_instruction", ""),
            max_length=int(manifest.get("max_length", 2048)),
            embedding_dimension=manifest.get("embedding_dimension"),
            base_url_env=manifest.get("base_url_env", "DASHSCOPE_BASE_URL"),
            api_key_env=manifest.get("api_key_env", "DASHSCOPE_API_KEY"),
            api_batch_size=int(manifest.get("api_batch_size", 20)),
        )
        with (source / "chunks.jsonl").open("r", encoding="utf-8") as handle:
            instance.chunks = [
                CaseChunk.model_validate_json(line) for line in handle if line.strip()
            ]
        with (source / "bm25.pkl").open("rb") as handle:
            instance.bm25 = pickle.load(handle)
        instance.vector_index = instance.faiss.read_index(str(source / "vectors.faiss"))
        if len(instance.chunks) != instance.vector_index.ntotal:
            raise ValueError("Chunk metadata and FAISS vector counts do not match")
        return instance
