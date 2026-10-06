"""Search documentation tool for the coding agent.

Provides vector and hybrid search capabilities over indexed documentation.
"""

from __future__ import annotations

import asyncio
from typing import Any


def search_docs(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    include_graph: bool = False,
    _context: Any = None,
    **_kwargs: Any,
) -> list[dict[str, Any]]:
    """Search indexed documentation from synchronous code.

    A thin wrapper over ``search_docs_async`` for callers that have no event
    loop. Async callers -- the agent loop included -- await
    ``search_docs_async`` instead: this drives a fresh loop and raises if one
    is already running.

    Args:
        query: Search query text
        top_k: Number of results to return
        mode: Search mode (vector, keyword, hybrid, all)
        include_graph: Include knowledge graph in search
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        List of search results with content, source, and score
    """
    return asyncio.run(
        search_docs_async(
            query=query,
            top_k=top_k,
            mode=mode,
            include_graph=include_graph,
            _context=_context,
            **_kwargs,
        )
    )


async def search_docs_async(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    include_graph: bool = False,
    _context: Any = None,
    **_kwargs: Any,
) -> list[dict[str, Any]]:
    """Search indexed documentation.

    This is the implementation; ``search_docs`` is the synchronous wrapper.

    Args:
        query: Search query text
        top_k: Number of results to return
        mode: Search mode (vector, keyword, hybrid, all)
        include_graph: Include knowledge graph in search
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        List of search results with content, source, and score
    """
    from docugraph.retrieval.hybrid_search import (
        FusionStrategy,
        HybridRetriever,
        HybridSearchConfig,
        SearchMode,
    )

    mode_map = {
        "vector": SearchMode.VECTOR,
        "keyword": SearchMode.KEYWORD,
        "hybrid": SearchMode.HYBRID,
        "all": SearchMode.ALL,
    }

    config = HybridSearchConfig(include_graph=include_graph)
    retriever = HybridRetriever(config=config)

    results = await retriever.asearch(
        query=query,
        top_k=top_k,
        mode=mode_map.get(mode, SearchMode.HYBRID),
        fusion=FusionStrategy.RRF,
    )

    # Format results
    formatted = []
    for result in results:
        chunk = result.chunk
        formatted.append(
            {
                "content": chunk.content,
                "source": chunk.metadata.get("source_url")
                or chunk.metadata.get("source_path", "Unknown"),
                "title": chunk.metadata.get("title", "Untitled"),
                "section": chunk.metadata.get("section_title", ""),
                "score": result.score,
                "sources": result.sources,
                "metadata": chunk.metadata,
            }
        )

    return formatted


def search_by_source(
    query: str,
    source_pattern: str,
    top_k: int = 10,
    _context: Any = None,
    **_kwargs: Any,
) -> list[dict[str, Any]]:
    """Search documentation filtered by source pattern.

    Args:
        query: Search query text
        source_pattern: Pattern to filter sources (e.g., "react", "fastapi")
        top_k: Number of results to return
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        List of filtered search results
    """
    # Get more results than needed for filtering
    results = search_docs(query=query, top_k=top_k * 3, _context=_context)

    # Filter by source pattern
    pattern_lower = source_pattern.lower()
    filtered = [
        r
        for r in results
        if pattern_lower in r.get("source", "").lower()
        or pattern_lower in r.get("title", "").lower()
    ]

    return filtered[:top_k]
