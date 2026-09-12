"""Integration test: GraphStore initializes offline with no cloud keys.

Regression coverage for three graphiti-core 0.30 wiring issues fixed in v1:
  1. Graphiti's default cross_encoder (OpenAIRerankerClient) demanded an
     OPENAI_API_KEY even for an Ollama/local setup -> we inject NoopCrossEncoder.
  2. graphiti-core 0.30 validates injected clients with pydantic is_instance_of,
     so the embedder/cross-encoder must subclass Graphiti's base classes.
  3. The deprecated KuzuDriver never sets `_database`, which
     Graphiti._resolve_request_scope reads for any non-default group_id.

This test builds a real Graphiti+Kuzu graph with a fake LLM client (no network)
and a fake fastembed embedder (no ONNX download), then adds an episode with a
non-default group_id to exercise the `_database` path.
"""

import pytest

from docugraph.core.embeddings import EmbeddingProviderBase


class FakeEmbedder(EmbeddingProviderBase):
    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed(self, texts):
        return [[0.1] * self._dim for _ in texts]

    def embed_query(self, query):  # noqa: ARG002 -- fixed vector regardless of query
        return [0.1] * self._dim

    @property
    def dimensions(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fake-graph-embedder"


def _make_fake_llm_client():
    """A minimal Graphiti LLMClient that never hits the network.

    Graphiti calls the private `_generate_response`; we return an empty
    extraction so add_episode completes without any LLM traffic.
    """
    from graphiti_core.llm_client.client import LLMClient
    from graphiti_core.llm_client.config import LLMConfig

    class FakeGraphitiLLM(LLMClient):
        def __init__(self):
            super().__init__(config=LLMConfig(model="fake", small_model="fake"))

        async def _generate_response(
            self,
            messages,  # noqa: ARG002 -- Graphiti-mandated signature
            response_model=None,
            max_tokens=None,  # noqa: ARG002
            model_size=None,  # noqa: ARG002
        ):
            # Return a schema-valid "nothing extracted" response. Graphiti
            # validates the dict against response_model, so populate required
            # list fields with empty lists and leave scalars at their defaults.
            if response_model is None:
                return {}
            result: dict = {}
            for field_name, field in response_model.model_fields.items():
                ann = field.annotation
                # Default list-typed fields to [] so validation passes.
                if ann is not None and "list" in str(ann).lower():
                    result[field_name] = []
            return result

    return FakeGraphitiLLM()


@pytest.mark.asyncio
async def test_graph_store_initializes_without_cloud_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    import docugraph.storage.graph_store as gs

    # Inject the fake LLM + fake embedder (no network, no ONNX download).
    monkeypatch.setattr(gs, "get_graphiti_llm_client", _make_fake_llm_client)
    monkeypatch.setattr(gs, "_get_embedder", lambda: gs.GraphitiEmbedderAdapter(FakeEmbedder()))

    store = gs.GraphStore(db_path=tmp_path / "graph")
    try:
        # Must not raise: proves NoopCrossEncoder + embedder subclass pass
        # Graphiti's is_instance_of validation and indices build offline.
        graphiti = await store._ensure_initialized()
        assert graphiti is not None
        # The _database workaround must be in place for non-default group_ids.
        assert hasattr(store._driver, "_database")

        # A non-default group_id would previously AttributeError in
        # _resolve_request_scope; with an empty extraction it completes.
        result = await store.add_episode(
            name="ep1",
            content="FastAPI depends on Starlette.",
            group_id="default",
        )
        assert "episode_uuid" in result
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_noop_cross_encoder_preserves_order():
    from docugraph.storage.graph_store import NoopCrossEncoder

    ranked = await NoopCrossEncoder().rank("q", ["a", "b", "c"])
    assert [p for p, _ in ranked] == ["a", "b", "c"]
    # Descending pseudo-scores.
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_noop_cross_encoder_empty():
    from docugraph.storage.graph_store import NoopCrossEncoder

    assert await NoopCrossEncoder().rank("q", []) == []
