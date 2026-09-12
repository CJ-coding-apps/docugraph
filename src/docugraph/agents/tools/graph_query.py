"""Knowledge graph query tool for the coding agent.

Provides access to the knowledge graph for entity and relationship queries.
"""

from __future__ import annotations

from typing import Any


async def graph_query(
    query: str,
    num_results: int = 10,
    group_id: str | None = None,
    _context: Any = None,
    **_kwargs: Any,
) -> list[dict[str, Any]]:
    """Query the knowledge graph for facts and relationships.

    Args:
        query: Natural language query
        num_results: Maximum number of results
        group_id: Optional group ID filter
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        List of facts from the knowledge graph
    """
    from docugraph.storage.graph_store import GraphStore

    store = GraphStore()
    group_ids = [group_id] if group_id else None

    try:
        results = await store.search(
            query=query,
            group_ids=group_ids,
            num_results=num_results,
        )
        return results
    finally:
        await store.close()


async def graph_add_content(
    content: str,
    name: str | None = None,
    group_id: str = "default",
    source_type: str = "text",
    _context: Any = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Add content to the knowledge graph.

    Args:
        content: Content to extract entities from
        name: Episode name (optional)
        group_id: Group ID for scoping
        source_type: Content type (text, json)
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        Information about extracted entities and relationships
    """
    from docugraph.storage.graph_store import GraphStore

    store = GraphStore()

    try:
        episode_name = name or f"episode_{hash(content) % 10000}"
        result = await store.add_episode(
            name=episode_name,
            content=content,
            source_type=source_type,
            group_id=group_id,
        )
        return result
    finally:
        await store.close()


async def find_related_entities(
    entity_name: str,
    relationship_type: str | None = None,
    max_depth: int = 2,
    _context: Any = None,
    **_kwargs: Any,
) -> list[dict[str, Any]]:
    """Find entities related to a given entity.

    Args:
        entity_name: Name of the entity to start from
        relationship_type: Filter by relationship type (USES, DEPENDS_ON, etc.)
        max_depth: How deep to look — widens the result pool per hop
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        List of related entities with relationship information
    """
    # Query for the entity and its relationships
    query = f"What is related to {entity_name}?"
    if relationship_type:
        query = f"What does {entity_name} {relationship_type.lower().replace('_', ' ')}?"

    # Deeper searches pull a proportionally wider pool of graph facts.
    results = await graph_query(
        query=query,
        num_results=10 * max(1, max_depth),
        _context=_context,
    )

    # Filter results that mention the entity
    related = []
    entity_lower = entity_name.lower()

    for r in results:
        fact = r.get("fact", "").lower()
        if entity_lower in fact:
            related.append(r)

    return related


async def get_entity_context(
    entity_name: str,
    include_relationships: bool = True,
    _context: Any = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Get comprehensive context about an entity.

    Args:
        entity_name: Name of the entity
        include_relationships: Include relationship information
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        Entity context including facts, relationships, and documentation
    """
    # Query for entity facts
    entity_facts = await graph_query(
        query=f"What is {entity_name}?",
        num_results=10,
        _context=_context,
    )

    context = {
        "entity": entity_name,
        "facts": entity_facts,
        "relationships": [],
        "documentation": [],
    }

    if include_relationships:
        # Get relationships
        relationships = await find_related_entities(
            entity_name=entity_name,
            _context=_context,
        )
        context["relationships"] = relationships

    # Also search documentation
    from docugraph.agents.tools.search_docs import search_docs

    doc_results = search_docs(
        query=entity_name,
        top_k=3,
        _context=_context,
    )
    context["documentation"] = doc_results

    return context


def query_entities_sync(
    query: str,
    num_results: int = 10,
    group_id: str | None = None,
    _context: Any = None,
    **_kwargs: Any,
) -> list[dict[str, Any]]:
    """Synchronous wrapper for graph_query.

    Args:
        query: Natural language query
        num_results: Maximum number of results
        group_id: Optional group ID filter
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        List of facts from the knowledge graph
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If in async context, create new event loop
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    graph_query(query, num_results, group_id, _context),
                )
                return future.result()
        else:
            return loop.run_until_complete(graph_query(query, num_results, group_id, _context))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(graph_query(query, num_results, group_id, _context))
