"""Unit tests for GraphitiEmbedderAdapter.

Verifies the async Graphiti EmbedderClient shim delegates to the injected
sync docugraph embedder with correct shapes.
"""

import pytest

from docugraph.core.embeddings import EmbeddingProviderBase
from docugraph.storage.graph_store import GraphitiEmbedderAdapter


class FakeEmbedder(EmbeddingProviderBase):
    def __init__(self, dim: int = 4):
        self._dim = dim
        self.query_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def embed(self, texts):
        self.batch_calls.append(list(texts))
        return [[float(i)] * self._dim for i, _ in enumerate(texts)]

    def embed_query(self, query):
        self.query_calls.append(query)
        return [1.0] * self._dim

    @property
    def dimensions(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fake"


@pytest.mark.asyncio
async def test_create_delegates_to_embed_query():
    embedder = FakeEmbedder(dim=4)
    adapter = GraphitiEmbedderAdapter(embedder=embedder)

    vector = await adapter.create("entity name")

    assert vector == [1.0, 1.0, 1.0, 1.0]
    assert embedder.query_calls == ["entity name"]


@pytest.mark.asyncio
async def test_create_batch_delegates_to_embed():
    embedder = FakeEmbedder(dim=3)
    adapter = GraphitiEmbedderAdapter(embedder=embedder)

    vectors = await adapter.create_batch(["a", "b"])

    assert len(vectors) == 2
    assert all(len(v) == 3 for v in vectors)
    assert embedder.batch_calls == [["a", "b"]]


@pytest.mark.asyncio
async def test_create_batch_empty():
    embedder = FakeEmbedder()
    adapter = GraphitiEmbedderAdapter(embedder=embedder)

    assert await adapter.create_batch([]) == []
    # Empty batch should short-circuit without calling the embedder.
    assert embedder.batch_calls == []
