"""Configuration management for DocuGraph AI."""

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    AUTO = "auto"
    FASTEMBED = "fastembed"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OLLAMA = "ollama"
    OPENAI = "openai"
    COHERE = "cohere"


class LLMProvider(str, Enum):
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

    provider=auto resolves: OpenAI (if OPENAI_API_KEY set and the openai
    package is installed) -> fastembed (local, pure ONNX, no torch)
    -> sentence-transformers (only if already installed, requires torch).
    """

    provider: EmbeddingProvider = EmbeddingProvider.AUTO
    model: str = "BAAI/bge-small-en-v1.5"  # fastembed default; used by auto/fastembed
    dimensions: int | None = None  # Auto-detected from model
    batch_size: int = 32


class LLMConfig(BaseModel):
    """LLM configuration for extraction and reranking."""

    provider: LLMProvider = LLMProvider.AUTO
    model: str = "auto"
    temperature: float = 0.0
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
    user_agent: str = "DocuGraph-AI/0.1.0"
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
