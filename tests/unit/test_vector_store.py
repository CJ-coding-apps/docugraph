"""Unit tests for VectorStore, focused on the embedding-model guard.

Uses fake embedders (fixed vectors) against a real temp LanceDB — no ONNX
weights are downloaded.
"""

import pytest

from docugraph.core.embeddings import EmbeddingProviderBase
from docugraph.core.models import Chunk
from docugraph.storage.vector_store import VectorStore


class FakeEmbedder(EmbeddingProviderBase):
    """Deterministic embedder with a configurable dimension and model name."""

    def __init__(self, model_name: str = "fake-model", dim: int = 4):
        self._model_name = model_name
        self._dim = dim

    def embed(self, texts):
        return [[float(i % 3)] * self._dim for i, _ in enumerate(texts)]

    def embed_query(self, query):  # noqa: ARG002 -- fixed vector regardless of query
        return [0.0] * self._dim

    @property
    def dimensions(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name


def _chunk(content: str) -> Chunk:
    return Chunk(document_id="doc1", content=content)


class TestVectorStoreRoundTrip:
    def test_add_and_search(self, tmp_path):
        store = VectorStore(db_path=tmp_path / "vec", embedder=FakeEmbedder())
        added = store.add_chunks([_chunk("alpha"), _chunk("beta")])
        assert added == 2
        assert store.count() == 2

        results = store.search("anything", top_k=5)
        assert len(results) == 2
        assert {r.chunk.content for r in results} == {"alpha", "beta"}

    def test_embedding_model_recorded(self, tmp_path):
        store = VectorStore(db_path=tmp_path / "vec", embedder=FakeEmbedder(model_name="model-a"))
        store.add_chunks([_chunk("alpha")])

        table = store._ensure_table()
        stored_model = table.to_arrow().column("embedding_model")[0].as_py()
        assert stored_model == "model-a"


class TestVectorStoreGuard:
    def test_dimension_mismatch_raises(self, tmp_path):
        # Build with a 4-dim embedder...
        store = VectorStore(db_path=tmp_path / "vec", embedder=FakeEmbedder(dim=4))
        store.add_chunks([_chunk("alpha")])

        # ...then reopen with a different dimension.
        store2 = VectorStore(db_path=tmp_path / "vec", embedder=FakeEmbedder(dim=8))
        with pytest.raises(RuntimeError, match="dim embeddings"):
            store2._ensure_table()

    def test_model_identity_mismatch_raises_even_when_dims_match(self, tmp_path):
        # Both embedders are 4-dim but different models — the 384/384 hazard.
        store = VectorStore(
            db_path=tmp_path / "vec", embedder=FakeEmbedder(model_name="model-a", dim=4)
        )
        store.add_chunks([_chunk("alpha")])

        store2 = VectorStore(
            db_path=tmp_path / "vec", embedder=FakeEmbedder(model_name="model-b", dim=4)
        )
        with pytest.raises(RuntimeError, match="docugraph clear"):
            store2._ensure_table()

    def test_same_model_reopen_ok(self, tmp_path):
        store = VectorStore(
            db_path=tmp_path / "vec", embedder=FakeEmbedder(model_name="model-a", dim=4)
        )
        store.add_chunks([_chunk("alpha")])

        store2 = VectorStore(
            db_path=tmp_path / "vec", embedder=FakeEmbedder(model_name="model-a", dim=4)
        )
        # Should not raise and should see the existing row.
        assert store2.count() == 1
