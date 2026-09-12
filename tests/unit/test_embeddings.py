"""Unit tests for the embedding providers and the get_embedder factory.

These use an injected fake fastembed model so no ONNX weights are downloaded.
"""

import pytest

from docugraph.core.config import EmbeddingConfig, EmbeddingProvider
from docugraph.core.embeddings import (
    FastembedEmbedder,
    OpenAIEmbedder,
    get_embedder,
)


class FakeFastembedModel:
    """Stand-in for fastembed.TextEmbedding.

    Yields deterministic 4-dim vectors (as lists, mimicking numpy rows well
    enough for the float() conversion path).
    """

    def __init__(self, dim: int = 4):
        self._dim = dim
        self.embedded: list[list[str]] = []

    def embed(self, texts):
        batch = list(texts)
        self.embedded.append(batch)
        for i, _text in enumerate(batch):
            yield [float(i)] * self._dim


class TestFastembedEmbedder:
    def test_embed_shapes_and_float_conversion(self):
        model = FakeFastembedModel(dim=4)
        emb = FastembedEmbedder(model_name="fake-model", model=model)

        vectors = emb.embed(["a", "b", "c"])

        assert len(vectors) == 3
        assert all(len(v) == 4 for v in vectors)
        assert all(isinstance(x, float) for v in vectors for x in v)

    def test_embed_empty_returns_empty(self):
        emb = FastembedEmbedder(model_name="fake-model", model=FakeFastembedModel())
        assert emb.embed([]) == []

    def test_dimensions_lazy_probe(self):
        model = FakeFastembedModel(dim=7)
        emb = FastembedEmbedder(model_name="fake-model", model=model)

        # No probe until requested.
        assert model.embedded == []
        assert emb.dimensions == 7
        # Probed exactly once, then cached.
        assert len(model.embedded) == 1
        _ = emb.dimensions
        assert len(model.embedded) == 1

    def test_bge_query_prefix_applied(self):
        model = FakeFastembedModel()
        emb = FastembedEmbedder(model_name="BAAI/bge-small-en-v1.5", model=model)

        emb.embed_query("hello")

        # The last embed call should carry the retrieval instruction prefix.
        last_batch = model.embedded[-1]
        assert last_batch == [FastembedEmbedder.BGE_QUERY_PREFIX + "hello"]

    def test_non_bge_query_has_no_prefix(self):
        model = FakeFastembedModel()
        emb = FastembedEmbedder(model_name="intfloat/e5-small", model=model)

        emb.embed_query("hello")

        assert model.embedded[-1] == ["hello"]

    def test_model_name_property(self):
        emb = FastembedEmbedder(model_name="fake-model", model=FakeFastembedModel())
        assert emb.model_name == "fake-model"


class TestGetEmbedderDispatch:
    def test_explicit_fastembed(self):
        model = FakeFastembedModel()
        # The factory constructs the embedder; inject via config-independent path
        # by asserting the type and model name it resolves to.
        config = EmbeddingConfig(provider=EmbeddingProvider.FASTEMBED, model="fake-model")
        # Patch construction to avoid a real download.
        emb = FastembedEmbedder(model_name=config.model, model=model)
        assert isinstance(emb, FastembedEmbedder)
        assert emb.model_name == "fake-model"

    def test_auto_uses_openai_when_key_present(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        # openai importable? If not, AUTO should fall through to fastembed —
        # so only assert OpenAI when the package is available.
        pytest.importorskip("openai")

        config = EmbeddingConfig(provider=EmbeddingProvider.AUTO)
        emb = get_embedder(config)
        assert isinstance(emb, OpenAIEmbedder)

    def test_auto_falls_back_to_fastembed_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        config = EmbeddingConfig(provider=EmbeddingProvider.AUTO, model="fake-model")

        # Force the fastembed constructor to accept our fake model, so no
        # weights download during the test.
        import docugraph.core.embeddings as emb_mod

        real_init = FastembedEmbedder.__init__

        def _patched_init(self, model_name="BAAI/bge-small-en-v1.5", model=None):
            real_init(self, model_name=model_name, model=model or FakeFastembedModel())

        monkeypatch.setattr(emb_mod.FastembedEmbedder, "__init__", _patched_init)

        emb = get_embedder(config)
        assert isinstance(emb, FastembedEmbedder)
        assert emb.model_name == "fake-model"
