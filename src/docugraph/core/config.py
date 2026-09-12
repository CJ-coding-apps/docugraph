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

    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OLLAMA = "ollama"
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"


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


class EmbeddingFallback(BaseModel):
    """Fallback embedding provider configuration."""

    provider: EmbeddingProvider
    model: str


class EmbeddingConfig(BaseModel):
    """Embedding configuration (local-first, LLM-agnostic)."""

    provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    model: str = "all-MiniLM-L6-v2"
    dimensions: int | None = None  # Auto-detected from model
    batch_size: int = 32
    device: str = "auto"  # auto, cpu, cuda, mps
    fallback: list[EmbeddingFallback] = Field(default_factory=list)

    def get_device(self) -> str:
        """Determine the best available device."""
        if self.device != "auto":
            return self.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"


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


class ServerConfig(BaseModel):
    """REST API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    debug: bool = False


class MCPConfig(BaseModel):
    """MCP server configuration."""

    transport: str = "stdio"  # stdio or http
    port: int = 3000


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
    server: ServerConfig = Field(default_factory=ServerConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

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
