from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from legalmind.retrieval.embedding import OpenAIEmbeddingBackend
from legalmind.retrieval.reranker import APIReranker
from legalmind.schemas import SearchHit


class FakeEmbeddings:
    def __init__(self) -> None:
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        data = [
            SimpleNamespace(index=index, embedding=[float(len(text)), 1.0])
            for index, text in reversed(list(enumerate(request["input"])))
        ]
        return SimpleNamespace(data=data)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.3},
            ]
        }


class FakeHTTPClient:
    def __init__(self) -> None:
        self.request = None

    def post(self, url, json):
        self.request = (url, json)
        return FakeResponse()


def make_hit(case_id: str, text: str, score: float) -> SearchHit:
    return SearchHit(
        case_id=case_id,
        chunk_id=f"{case_id}-0",
        text=text,
        accusations=[],
        relevant_articles=[],
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        score=score,
    )


def test_openai_embedding_backend_batches_orders_and_normalizes() -> None:
    client = FakeEmbeddingClient()
    backend = OpenAIEmbeddingBackend(
        "qwen3.7-text-embedding",
        "UNUSED_BASE_URL",
        "UNUSED_API_KEY",
        dimensions=2,
        api_batch_size=2,
        client=client,
    )
    values = backend.encode_documents(["a", "abcd", "xy"], batch_size=8)

    assert values.shape == (3, 2)
    assert len(client.embeddings.requests) == 2
    assert client.embeddings.requests[0]["dimensions"] == 2
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), np.ones(3), atol=1e-6)
    assert values[0, 0] < values[1, 0]


def test_api_reranker_maps_service_indices(monkeypatch) -> None:
    monkeypatch.setenv("RERANK_URL", "https://example.test/reranks")
    client = FakeHTTPClient()
    reranker = APIReranker(
        "qwen3-rerank",
        "RERANK_URL",
        "UNUSED_API_KEY",
        instruction="legal relevance",
        client=client,
    )
    hits = [make_hit("a", "first", 0.1), make_hit("b", "second", 0.2)]
    results = reranker.rerank("query", hits, top_k=2)

    assert [item.case_id for item in results] == ["b", "a"]
    assert results[0].source_scores == {"rrf": 0.2, "reranker": 0.9}
    assert client.request[1]["instruct"] == "legal relevance"
