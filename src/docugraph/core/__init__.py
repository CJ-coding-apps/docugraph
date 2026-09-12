"""Core configuration, models, and provider abstractions."""

from docugraph.core.config import Config, get_config
from docugraph.core.embeddings import EmbeddingProviderBase, get_embedder
from docugraph.core.llm import (
    LLMProviderBase,
    get_graphiti_llm_client,
    get_llm_client,
    get_llm_provider,
    is_ollama_available,
)
from docugraph.core.models import Chunk, Document, Entity, MemoryEntry, Relationship, SearchResult

__all__ = [
    # Config
    "Config",
    "get_config",
    # Models
    "Document",
    "Chunk",
    "Entity",
    "Relationship",
    "MemoryEntry",
    "SearchResult",
    # Embeddings
    "EmbeddingProviderBase",
    "get_embedder",
    # LLM
    "LLMProviderBase",
    "get_llm_provider",
    "get_llm_client",
    "get_graphiti_llm_client",
    "is_ollama_available",
]
