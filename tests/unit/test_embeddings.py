"""Unit tests for the embedding providers and the get_embedder factory.

These use an injected fake fastembed model so no ONNX weights are downloaded.
"""

import pytest

from docugraph.core.config import EmbeddingConfig, EmbeddingProvider
from docugraph.core.embeddings import (
    CohereEmbedder,
    FastembedEmbedder,
    OllamaEmbedder,
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


def _fake_fastembed_init():
    """A FastembedEmbedder.__init__ that takes the fake model, not real weights.

    Captured before monkeypatch swaps it in, so the real constructor still runs
    and only the model is replaced. Without this, AUTO would download ~130 MB of
    ONNX weights during the test.
    """
    real_init = FastembedEmbedder.__init__

    def _init(self, model_name="BAAI/bge-small-en-v1.5", model=None):
        real_init(self, model_name=model_name, model=model or FakeFastembedModel())

    return _init


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

    def test_auto_stays_local_even_with_a_cloud_key_present(self, monkeypatch):
        """A key in the environment is not a request to send documents anywhere.

        `openai` is importable here (the CI test job installs the `cloud` extra)
        and the key is set, so under the old resolution order this returned an
        OpenAIEmbedder and every document embedded went to OpenAI with nothing
        in the config asking for it. The assert on the type is the whole test:
        if anything in AUTO consults the environment again, this fails.

        The importorskip is what kept this test from running at all in CI for a
        while: the test job installed `--extra dev` only, and the one job that
        did install the SDKs ran a single other file. A guard that never runs
        reads as a passing test.
        """
        pytest.importorskip("openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        import docugraph.core.embeddings as emb_mod

        monkeypatch.setattr(emb_mod.FastembedEmbedder, "__init__", _fake_fastembed_init())

        emb = get_embedder(EmbeddingConfig(provider=EmbeddingProvider.AUTO))
        assert isinstance(emb, FastembedEmbedder)

    def test_auto_uses_the_local_backend(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        config = EmbeddingConfig(provider=EmbeddingProvider.AUTO, model="fake-model")

        import docugraph.core.embeddings as emb_mod

        monkeypatch.setattr(emb_mod.FastembedEmbedder, "__init__", _fake_fastembed_init())

        emb = get_embedder(config)
        assert isinstance(emb, FastembedEmbedder)
        assert emb.model_name == "fake-model"

    def test_naming_openai_still_sends_documents_there(self):
        """The explicit provider is the request, and it is unaffected."""
        pytest.importorskip("openai")
        emb = get_embedder(
            EmbeddingConfig(provider=EmbeddingProvider.OPENAI, model="text-embedding-x")
        )
        assert isinstance(emb, OpenAIEmbedder)
        assert emb.model_name == "text-embedding-x"

    def test_a_cloud_embedding_provider_without_a_model_is_an_error(self):
        """Naming the provider is not the same as naming the model.

        The two cloud embedders used to carry a default, both wrong in the same
        way: OpenAI's table answered 1536 for anything it did not list, and
        Cohere's was the literal 1024 that belongs to embed-english-v3.0 alone.
        A default that is right only for one model is a wrong answer for the
        rest, so there is no default and the failure names the setting.
        """
        with pytest.raises(ValueError, match="embeddings.model must name"):
            OpenAIEmbedder(model_name=None)

        with pytest.raises(ValueError, match="embeddings.model must name"):
            CohereEmbedder(model_name=None)

    def test_local_backends_keep_their_own_defaults(self):
        """Local is where a default is a convenience rather than a guess.

        Ollama is reachable without an account and its model is a `ollama pull`
        away, so a name here is a fact the reader can check. This is the other
        half of the test above: the carve-out is deliberate, and it is exactly
        the local backends.
        """
        local = [
            FastembedEmbedder(model_name=None, model=FakeFastembedModel()).model_name,
            OllamaEmbedder(model_name=None).model_name,
        ]
        assert local == ["BAAI/bge-small-en-v1.5", "nomic-embed-text"]
