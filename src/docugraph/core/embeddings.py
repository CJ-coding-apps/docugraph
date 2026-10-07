"""Embedding provider abstraction (local-first, LLM-agnostic)."""

from abc import ABC, abstractmethod
from typing import Any

from docugraph.core.config import EmbeddingConfig, EmbeddingProvider, get_config


class EmbeddingProviderBase(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query (may use different model/prefix)."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...


class FastembedEmbedder(EmbeddingProviderBase):
    """Local embeddings via fastembed (pure ONNX, no torch).

    Default model BAAI/bge-small-en-v1.5 (384 dims). Weights download on
    first use and are cached under FASTEMBED_CACHE_PATH; unset, fastembed
    falls back to a `fastembed_cache` subdirectory of the system temp
    directory (its `define_cache_dir`), not to `~/.cache/fastembed`.
    """

    # bge-en-v1.5 models are trained with a query/passage asymmetry; retrieval
    # queries should carry this instruction prefix for best relevance.
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str | None = None, model: Any | None = None):
        self._model_name = model_name or "BAAI/bge-small-en-v1.5"
        self._dimensions: int | None = None
        if model is None:
            # Lazy import so environments without fastembed degrade via the chain.
            from fastembed import TextEmbedding

            model = TextEmbedding(self._model_name)
        # Stored only once non-None so mypy can narrow the attribute.
        self._model: Any = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts (passages/documents)."""
        if not texts:
            return []
        # fastembed yields numpy arrays; convert to plain float lists.
        vectors: list[list[float]] = [[float(x) for x in vec] for vec in self._model.embed(texts)]
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a retrieval query, applying the bge instruction prefix."""
        prefixed = self.BGE_QUERY_PREFIX + query if "bge" in self._model_name.lower() else query
        return self.embed([prefixed])[0]

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._dimensions = len(self.embed(["dimension probe"])[0])
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name


class SentenceTransformersEmbedder(EmbeddingProviderBase):
    """Local embeddings using sentence-transformers (optional; requires torch).

    Not a managed dependency anymore — works if the user has
    sentence-transformers installed independently.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self._model_name = model_name or "all-MiniLM-L6-v2"
        self._batch_size = batch_size

        # Lazy load to avoid import overhead
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name, device=device)
        detected = self._model.get_sentence_embedding_dimension()
        self._dimensions: int = int(detected) if detected is not None else 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        vectors: list[list[float]] = embeddings.tolist()
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        embedding = self._model.encode(query, convert_to_numpy=True)
        vector: list[float] = embedding.tolist()
        return vector

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name


class OllamaEmbedder(EmbeddingProviderBase):
    """Embeddings via Ollama (local LLM server)."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str = "http://localhost:11434",
    ):
        # Local, so it keeps a default: the server is on this machine and
        # `ollama pull nomic-embed-text` is a command the reader can run.
        self._model_name = model_name or "nomic-embed-text"
        self._base_url = base_url
        self._dimensions: int | None = None

    def _get_client(self) -> Any:
        """Get Ollama client."""
        try:
            import ollama

            return ollama.Client(host=self._base_url)
        except ImportError as e:
            raise ImportError(
                "Ollama support requires the 'ollama' package. "
                "Install with: pip install 'docugraph[ollama]'"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        client = self._get_client()
        embeddings = []
        for text in texts:
            response = client.embeddings(model=self._model_name, prompt=text)
            embeddings.append(response["embedding"])
            if self._dimensions is None:
                self._dimensions = len(response["embedding"])
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self.embed([query])[0]

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # Get dimensions by embedding a test string
            self.embed(["test"])
        return self._dimensions or 768

    @property
    def model_name(self) -> str:
        return self._model_name


class OpenAIEmbedder(EmbeddingProviderBase):
    """Embeddings via OpenAI API.

    ``model_name`` has no default. An embedding model's vectors live in their
    own space and cannot be mixed with another's, so a name compiled in here is
    a name that silently becomes wrong the day OpenAI retires it -- and the
    store's dimension and identity guards would then be comparing against a
    model nobody chose.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
    ):
        if not model_name:
            raise ValueError(
                "embeddings.model must name an OpenAI embedding model when "
                "embeddings.provider is 'openai' (e.g. DOCUGRAPH_EMBEDDINGS__MODEL). "
                "There is no default: an embedding model fixes the vector space "
                "of the whole index, so it is not something to guess at."
            )
        self._model_name = model_name
        self._api_key = api_key
        # Measured from a real response rather than looked up in a table. The
        # table was a second place to keep a fact the API already reports, and
        # it was wrong for every model it did not list -- it defaulted to 1536,
        # so an unlisted model got a schema that could not hold its vectors.
        self._dimensions: int | None = None

    def _get_client(self) -> Any:
        """Get OpenAI client."""
        try:
            from openai import OpenAI

            api_key = self._api_key
            if api_key is None:
                config = get_config()
                api_key = config.llm.openai_api_key

            if not api_key:
                raise ValueError("OpenAI API key not configured")

            return OpenAI(api_key=api_key)
        except ImportError as e:
            raise ImportError(
                "OpenAI support requires the 'openai' package. "
                "Install with: pip install 'docugraph[cloud]'"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        if self._dimensions is None and vectors:
            self._dimensions = len(vectors[0])
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self.embed([query])[0]

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # Nothing has been embedded yet and the caller needs the width before
            # it can build a schema, so ask the model directly.
            self.embed(["dimension probe"])
        return self._dimensions or 0

    @property
    def model_name(self) -> str:
        return self._model_name


class CohereEmbedder(EmbeddingProviderBase):
    """Embeddings via Cohere API.

    ``model_name`` has no default, for the same reason as OpenAIEmbedder: the
    vector space is a property of the model. It used to default to
    ``embed-english-v3.0`` with a hard-coded ``1024`` dimensions, which is that
    one model's width -- every other Cohere embedding model got a schema built
    to the wrong size.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
    ):
        if not model_name:
            raise ValueError(
                "embeddings.model must name a Cohere embedding model when "
                "embeddings.provider is 'cohere' (e.g. DOCUGRAPH_EMBEDDINGS__MODEL). "
                "There is no default: the model fixes the vector space of the "
                "whole index."
            )
        self._model_name = model_name
        self._api_key = api_key
        self._dimensions: int | None = None

    def _get_client(self) -> Any:
        """Get Cohere client."""
        try:
            import cohere

            api_key = self._api_key
            if api_key is None:
                import os

                api_key = os.getenv("COHERE_API_KEY")

            if not api_key:
                raise ValueError("Cohere API key not configured")

            return cohere.Client(api_key)
        except ImportError as e:
            raise ImportError(
                "Cohere support requires the 'cohere' package. "
                "Install with: pip install 'docugraph[cloud]'"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        client = self._get_client()
        response = client.embed(
            texts=texts,
            model=self._model_name,
            input_type="search_document",
        )
        vectors = [list(emb) for emb in response.embeddings]
        if self._dimensions is None and vectors:
            self._dimensions = len(vectors[0])
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        client = self._get_client()
        response = client.embed(
            texts=[query],
            model=self._model_name,
            input_type="search_query",
        )
        return list(response.embeddings[0])

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self.embed(["dimension probe"])
        return self._dimensions or 0

    @property
    def model_name(self) -> str:
        return self._model_name


def get_embedder(config: EmbeddingConfig | None = None) -> EmbeddingProviderBase:
    """Factory function to get the configured embedder.

    With provider=auto: fastembed (local, pure ONNX) -> sentence-transformers
    (only if already installed). Nothing about the environment is consulted,
    so an unrelated OPENAI_API_KEY in your shell cannot start sending document
    text to a third party. A cloud provider is used only when it is named
    explicitly, and then it fails loudly on a missing package or key.
    """
    if config is None:
        config = get_config().embeddings

    provider = config.provider

    if provider == EmbeddingProvider.FASTEMBED:
        return FastembedEmbedder(model_name=config.model)
    elif provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
        return SentenceTransformersEmbedder(
            model_name=config.model,
            batch_size=config.batch_size,
        )
    elif provider == EmbeddingProvider.OLLAMA:
        return OllamaEmbedder(model_name=config.model)
    elif provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbedder(model_name=config.model)
    elif provider == EmbeddingProvider.COHERE:
        return CohereEmbedder(model_name=config.model)
    else:
        # AUTO: local backends only, and `os.environ` is deliberately not
        # consulted here. A key in the environment is not a request to use it:
        # resolving to OpenAI on the mere presence of OPENAI_API_KEY meant that
        # a user who happened to have the variable exported (or the `cloud`
        # extra installed) had the text of every document they embedded sent to
        # a third party, with nothing in the config saying so. Naming the
        # provider is the request; `provider: openai` sends it there.
        try:
            return FastembedEmbedder(model_name=config.model)
        except Exception:  # nosec B110 -- provider chain: try the next backend
            pass
        try:
            return SentenceTransformersEmbedder(
                model_name=config.model,
                batch_size=config.batch_size,
            )
        except ImportError as e:
            raise RuntimeError(
                "No embedding provider available. Install fastembed "
                "(pip install fastembed) for local ONNX embeddings, or set "
                "embeddings.provider to 'openai' or 'cohere' for a cloud one."
            ) from e
