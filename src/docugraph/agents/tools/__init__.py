"""Agent tools for the coding agent.

Provides modular tools for:
- Documentation search (vector and hybrid)
- Code search with language awareness
- Knowledge graph queries
- Session memory operations
"""

from docugraph.agents.tools.graph_query import (
    find_related_entities,
    get_entity_context,
    graph_add_content,
    graph_query,
    query_entities_sync,
)
from docugraph.agents.tools.memory_ops import (
    memory_delete,
    memory_recall,
    memory_search,
    memory_store,
    recall_context,
    recall_decisions,
    store_context,
    store_decision,
)
from docugraph.agents.tools.search_code import (
    search_code,
    search_symbols,
)
from docugraph.agents.tools.search_docs import (
    search_by_source,
    search_docs,
    search_docs_async,
)

__all__ = [
    # Search docs
    "search_docs",
    "search_docs_async",
    "search_by_source",
    # Search code
    "search_code",
    "search_symbols",
    # Graph query
    "graph_query",
    "graph_add_content",
    "find_related_entities",
    "get_entity_context",
    "query_entities_sync",
    # Memory ops
    "memory_store",
    "memory_recall",
    "memory_delete",
    "memory_search",
    "store_decision",
    "recall_decisions",
    "store_context",
    "recall_context",
]
