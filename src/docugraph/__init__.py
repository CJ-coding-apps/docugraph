"""
DocuGraph AI - Intelligent coding assistant with knowledge graphs and vector search.

A powerful coding agent platform combining:
- Knowledge graphs (Graphiti + Kuzu)
- Vector search (LanceDB)
- Web crawling (Crawl4AI)
- Agent memory (MCP-compatible)
"""

__version__ = "0.1.0"

from docugraph.core.config import Config, get_config
from docugraph.core.models import Chunk, Document, Entity, MemoryEntry, Relationship

__all__ = [
    "__version__",
    "Config",
    "get_config",
    "Document",
    "Chunk",
    "Entity",
    "Relationship",
    "MemoryEntry",
]
