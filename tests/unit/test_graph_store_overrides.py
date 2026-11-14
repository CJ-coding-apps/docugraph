"""The keyword-only client overrides on ``GraphStore`` (Q1).

The benchmark opens a graph with an injected LLM client and embedder so it can
prove "no LLM at query time" and reuse the fixture's own embedder. That must not
change what an ordinary ``GraphStore()`` does: with no overrides it resolves the
configured clients exactly as before, and those are the only configured lookups.
"""

import pytest

from docugraph.core.embeddings import EmbeddingProviderBase
from docugraph.storage import graph_store as gs


class _Embedder(EmbeddingProviderBase):
    def __init__(self, name="fixture-embedder", dimensions=4):
        self._name = name
        self._dimensions = dimensions

    def embed(self, texts):
        return [[1.0] + [0.0] * (self._dimensions - 1) for _ in texts]

    def embed_query(self, query):
        return [1.0] + [0.0] * (self._dimensions - 1)

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def model_name(self):
        return self._name


def test_default_construction_resolves_the_configured_clients(monkeypatch, tmp_path):
    """No overrides: both resolvers return what they always did."""
    configured_llm = object()
    configured_embedder = object()
    calls = []

    def fake_llm():
        calls.append("llm")
        return configured_llm

    def fake_embedder():
        calls.append("embedder")
        return configured_embedder

    monkeypatch.setattr(gs, "get_graphiti_llm_client", fake_llm)
    monkeypatch.setattr(gs, "_get_embedder", fake_embedder)

    store = gs.GraphStore(tmp_path / "graph")

    assert store._resolve_llm_client() is configured_llm
    assert store._resolve_embedder() is configured_embedder
    assert calls == ["llm", "embedder"]


def test_injected_clients_are_used_and_config_is_not_consulted(monkeypatch, tmp_path):
    """Overrides win, and the configured lookups are not called at all."""

    def fail(*_args, **_kwargs):
        raise AssertionError("the configured client was resolved despite an override")

    monkeypatch.setattr(gs, "get_graphiti_llm_client", fail)
    monkeypatch.setattr(gs, "_get_embedder", fail)

    llm = object()
    embedder = _Embedder()
    store = gs.GraphStore(tmp_path / "graph", llm_client=llm, embedder=embedder)

    assert store._resolve_llm_client() is llm
    # A docugraph provider is adapted for Graphiti; the adapter wraps that
    # exact instance, so the store and the bench share one embedder.
    adapted = store._resolve_embedder()
    assert isinstance(adapted, gs.GraphitiEmbedderAdapter)
    assert adapted._embedder is embedder


def test_a_graphiti_embedder_is_passed_through_unchanged(tmp_path):
    """An already-Graphiti client is not wrapped (or rebuilt) a second time."""
    pytest.importorskip("graphiti_core")
    from graphiti_core.embedder.client import EmbedderClient

    class GraphitiEmbedder(EmbedderClient):
        async def create(self, input_data):
            return [0.0, 1.0]

        async def create_batch(self, input_data_list):
            return [[0.0, 1.0] for _ in input_data_list]

    embedder = GraphitiEmbedder()
    store = gs.GraphStore(tmp_path / "graph", embedder=embedder)
    assert store._resolve_embedder() is embedder


def test_overrides_do_not_leak_between_stores(monkeypatch, tmp_path):
    """One injected store does not change the next store's resolution."""
    configured = object()
    monkeypatch.setattr(gs, "get_graphiti_llm_client", lambda: configured)
    monkeypatch.setattr(gs, "_get_embedder", lambda: configured)

    injected = object()
    gs.GraphStore(tmp_path / "injected", llm_client=injected)
    plain = gs.GraphStore(tmp_path / "plain")
    assert plain._resolve_llm_client() is configured
