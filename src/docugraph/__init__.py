"""
DocuGraph AI - Intelligent coding assistant with knowledge graphs and vector search.

A powerful coding agent platform combining:
- Knowledge graphs (Graphiti + Kuzu)
- Vector search (LanceDB)
- Web crawling (Crawl4AI)
- Agent memory (MCP-compatible)

The names below are the supported Python surface. Some of them live in modules
that import an optional dependency at module level; importing this package must
stay cheap, so those are resolved on first access instead of eagerly.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from docugraph._optional import MissingExtraError
from docugraph._version import __version__
from docugraph.core.config import Config, get_config
from docugraph.core.models import Chunk, Document, Entity, MemoryEntry, Relationship
from docugraph.security.url_policy import UrlPolicy

if TYPE_CHECKING:
    from docugraph.ingestion.chunker import Chunker
    from docugraph.ingestion.crawler import DocCrawler
    from docugraph.retrieval.hybrid_search import FusionStrategy, HybridRetriever, SearchMode
    from docugraph.storage.vector_store import VectorStore

__all__ = [
    "__version__",
    "Config",
    "get_config",
    "Document",
    "Chunk",
    "Entity",
    "Relationship",
    "MemoryEntry",
    "DocCrawler",
    "Chunker",
    "VectorStore",
    "HybridRetriever",
    "SearchMode",
    "FusionStrategy",
    "UrlPolicy",
    "MissingExtraError",
]

# Name -> defining module, for the exports that are resolved on first access.
_LAZY: dict[str, str] = {
    "DocCrawler": "docugraph.ingestion.crawler",
    "Chunker": "docugraph.ingestion.chunker",
    "VectorStore": "docugraph.storage.vector_store",
    "HybridRetriever": "docugraph.retrieval.hybrid_search",
    "SearchMode": "docugraph.retrieval.hybrid_search",
    "FusionStrategy": "docugraph.retrieval.hybrid_search",
}


def __getattr__(name: str) -> Any:
    """Resolve a lazy export on first access and cache it in the namespace.

    Caching matters: a second ``from docugraph import DocCrawler`` must return
    the same object the first one did, not a freshly imported copy.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List everything the package advertises, resolved or not."""
    return sorted(set(globals()) | set(__all__))
