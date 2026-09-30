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
    first use and are cached under FASTEMBED_CACHE_PATH (default
    ~/.cache/fastembed).
    """

    # bge-en-v1.5 models are trained with a query/passage asymmetry; retrieval
    # queries should carry this instruction prefix for best relevance.
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", model: Any | None = None):
        self._model_name = model_name
        self._dimensions: int | None = None
        if model is None:
            # Lazy import so environments without fastembed degrade via the chain.
            from fastembed import TextEmbedding

            model = TextEmbedding(model_name)
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
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self._model_name = model_name
        self._batch_size = batch_size

        # Lazy load to avoid import overhead
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
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
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self._model_name = model_name
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
                "Install with: pip install 'docugraph-ai-v1[ollama]'"
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
    """Embeddings via OpenAI API."""

    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        self._model_name = model_name
        self._api_key = api_key

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
                "Install with: pip install 'docugraph-ai-v1[cloud]'"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self.embed([query])[0]

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS.get(self._model_name, 1536)

    @property
    def model_name(self) -> str:
        return self._model_name


class CohereEmbedder(EmbeddingProviderBase):
    """Embeddings via Cohere API."""

    def __init__(
        self,
        model_name: str = "embed-english-v3.0",
        api_key: str | None = None,
    ):
        self._model_name = model_name
        self._api_key = api_key
        self._dimensions = 1024  # Default for embed-english-v3.0

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
                "Install with: pip install 'docugraph-ai-v1[cloud]'"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        client = self._get_client()
        response = client.embed(
            texts=texts,
            model=self._model_name,
            input_type="search_document",
        )
        return [list(emb) for emb in response.embeddings]

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
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name


def get_embedder(config: EmbeddingConfig | None = None) -> EmbeddingProviderBase:
    """Factory function to get the configured embedder.

    Local-first priority. With provider=auto: OpenAI (if OPENAI_API_KEY is
    set and the openai package is installed) -> fastembed (local, pure ONNX)
    -> sentence-transformers (only if already installed). Explicit providers
    construct directly and fail loudly on missing packages/keys.
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
        # AUTO: resolve the best available backend (CF-v1 try-chain idiom).
        import os

        if os.environ.get("OPENAI_API_KEY"):
            try:
                import openai  # noqa: F401

                return OpenAIEmbedder(model_name="text-embedding-3-small")
            except ImportError:
                pass
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
                "OPENAI_API_KEY with the 'cloud' extra for OpenAI embeddings."
            ) from e
