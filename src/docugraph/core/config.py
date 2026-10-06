"""Configuration management for DocuGraph AI."""

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from docugraph import _version


class EmbeddingProvider(StrEnum):
    """Supported embedding providers."""

    AUTO = "auto"
    FASTEMBED = "fastembed"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OLLAMA = "ollama"
    OPENAI = "openai"
    COHERE = "cohere"


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    AUTO = "auto"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class StorageConfig(BaseModel):
    """Storage configuration."""

    data_dir: Path = Field(default=Path.home() / ".docugraph" / "data")
    vector_db: str = "lancedb"
    graph_db: str = "kuzu"

    def model_post_init(self, __context: Any) -> None:
        """Ensure data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


class EmbeddingConfig(BaseModel):
    """Embedding configuration (local-first, LLM-agnostic).

    provider=auto resolves: fastembed (local, pure ONNX, no torch) ->
    sentence-transformers (only if already installed, requires torch). It
    reads nothing from the environment, so a cloud provider is used only when
    one is named here.

    ``model`` has no default, because the backends do not share one. Each local
    backend names its own model in its constructor (fastembed's bge-small,
    sentence-transformers' MiniLM, Ollama's nomic-embed-text); a *cloud* backend
    has none, so leaving this unset with ``provider: openai`` or ``cohere`` is an
    error rather than a guess. That distinction matters: a single shared default
    here was the local fastembed name, so naming a cloud provider without naming
    a model would have sent ``BAAI/bge-small-en-v1.5`` to an API that has never
    heard of it.
    """

    provider: EmbeddingProvider = EmbeddingProvider.AUTO
    model: str | None = None  # local backends have their own; cloud has none
    dimensions: int | None = None  # measured from the model unless set here
    batch_size: int = 32


class LLMConfig(BaseModel):
    """LLM configuration for extraction and reranking.

    ``model`` defaults to the ``"auto"`` sentinel, which means "whichever model
    this provider offers". Only a local Ollama server can answer that, because
    it is the only provider docugraph can ask what it has installed. A cloud
    provider needs the model named: every name that could be written here has a
    retirement date on the provider's calendar, and a retired name fails every
    call with a 404 the user cannot act on.

    ``temperature`` is unset by default and is sent only when set. That is not
    tidiness: OpenAI's reasoning models reject any temperature other than their
    fixed default, so a default of 0.0 here made every call to one fail with
    ``unsupported value``. Set ``temperature: 0.0`` explicitly for deterministic
    local output.
    """

    provider: LLMProvider = LLMProvider.AUTO
    model: str = "auto"
    # Graphiti asks for two models: a large one for extraction and a small one
    # for the cheaper summarization and edge-dedup passes. Unset means "the same
    # as `model`, whatever that turns out to be" -- not a hard-coded model name,
    # which is what it was.
    small_model: str | None = None
    temperature: float | None = None
    max_tokens: int = 4096

    # API keys (loaded from environment)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    def model_post_init(self, __context: Any) -> None:
        """Load API keys from environment."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


class CrawlerConfig(BaseModel):
    """Web crawler configuration."""

    max_concurrent: int = 5
    rate_limit: float = 2.0  # requests per second
    user_agent: str = f"DocuGraph-AI/{_version.__version__}"
    respect_robots: bool = True
    cache_ttl: int = 86400  # 24 hours
    timeout: int = 30  # seconds


class Config(BaseSettings):
    """Main configuration for DocuGraph AI."""

    model_config = SettingsConfigDict(
        env_prefix="DOCUGRAPH_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    storage: StorageConfig = Field(default_factory=StorageConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from default locations."""
        # Check for config file in order of precedence
        config_paths = [
            Path.cwd() / "docugraph.yaml",
            Path.cwd() / "docugraph.yml",
            Path.cwd() / ".docugraph.yaml",
            Path.home() / ".docugraph" / "config.yaml",
            Path.home() / ".config" / "docugraph" / "config.yaml",
        ]

        for path in config_paths:
            if path.exists():
                return cls.from_yaml(path)

        return cls()

    def save(self, path: Path | None = None) -> None:
        """Save configuration to a YAML file."""
        if path is None:
            path = Path.home() / ".docugraph" / "config.yaml"

        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(mode="json")
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


@lru_cache
def get_config() -> Config:
    """Get the singleton configuration instance."""
    return Config.load()


def reset_config() -> None:
    """Reset the cached configuration (useful for testing)."""
    get_config.cache_clear()
