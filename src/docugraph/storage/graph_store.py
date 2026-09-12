"""Graph store backed by Graphiti + Kuzu for knowledge graph operations.

Supports LLM-agnostic configuration with local-first priority:
- Ollama (local, default if available)
- OpenAI (cloud)
- Anthropic (cloud)

Embeddings reuse the docugraph embedder (fastembed by default) via
GraphitiEmbedderAdapter, so graph ingestion needs no embedding service —
only the entity-extraction LLM (Graphiti design).

Uses the standalone LLM abstraction from docugraph.core.llm.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docugraph.core.config import EmbeddingProvider, get_config
from docugraph.core.embeddings import EmbeddingProviderBase, get_embedder
from docugraph.core.llm import get_graphiti_llm_client
from docugraph.core.models import Entity, EntityType, Relationship, RelationshipType


class GraphitiEmbedderAdapter:
    """Duck-typed Graphiti EmbedderClient backed by a docugraph embedder.

    Lets the doc knowledge graph reuse the same embedding backend as the
    vector store (fastembed by default) instead of requiring a separate
    Ollama/OpenAI embedding service. Implements Graphiti's async
    ``create``/``create_batch`` interface; the sync docugraph embedder runs
    in a worker thread.
    """

    def __init__(self, embedder: EmbeddingProviderBase | None = None):
        self._embedder = embedder or get_embedder()

    async def create(self, input_data: str) -> list[float]:
        """Embed a single text (Graphiti uses this for entity names)."""
        return await asyncio.to_thread(self._embedder.embed_query, input_data)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if not input_data_list:
            return []
        return await asyncio.to_thread(self._embedder.embed, input_data_list)


def _get_embedder() -> Any:
    """Create an embedder based on configuration.

    Ollama/OpenAI use Graphiti's own service-backed embedder; every other
    provider (fastembed default, sentence-transformers, cohere, auto)
    reuses the docugraph embedder via GraphitiEmbedderAdapter.

    Returns:
        Graphiti-compatible embedder
    """
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

    emb_config = get_config().embeddings

    if emb_config.provider == EmbeddingProvider.OLLAMA:
        embedder_config = OpenAIEmbedderConfig(
            api_key="ollama",
            embedding_model=emb_config.model or "nomic-embed-text",
            embedding_dim=emb_config.dimensions or 768,
            base_url="http://localhost:11434/v1",
        )
        return OpenAIEmbedder(config=embedder_config)

    elif emb_config.provider == EmbeddingProvider.OPENAI:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required for OpenAI embeddings")

        embedder_config = OpenAIEmbedderConfig(
            api_key=api_key,
            embedding_model=emb_config.model or "text-embedding-3-small",
            embedding_dim=emb_config.dimensions or 1536,
        )
        return OpenAIEmbedder(config=embedder_config)

    else:
        # fastembed / sentence-transformers / cohere / auto: reuse the
        # docugraph embedder — no embedding service required.
        return GraphitiEmbedderAdapter()


class GraphStore:
    """Knowledge graph store backed by Graphiti with Kuzu driver.

    Provides entity and relationship storage with temporal tracking,
    semantic search, and graph traversal capabilities.

    LLM-agnostic: Works with Ollama (local), OpenAI, or Anthropic.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize the graph store.

        Args:
            db_path: Path to the Kuzu database directory. If None, uses config default.
        """
        if db_path is None:
            config = get_config()
            db_path = config.storage.data_dir / "graph"

        self._db_path = Path(db_path)
        self._db_path.mkdir(parents=True, exist_ok=True)

        self._graphiti: Any | None = None
        self._driver: Any | None = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazily initialize Graphiti with Kuzu driver and configured LLM/embedder."""
        if self._initialized:
            return

        try:
            from graphiti_core import Graphiti
            from graphiti_core.driver.kuzu_driver import KuzuDriver
        except ImportError as e:
            raise ImportError(
                "graphiti-core[kuzu] is required for GraphStore. "
                "Install with: pip install 'graphiti-core[kuzu]'"
            ) from e

        # Create Kuzu driver
        kuzu_db_path = str(self._db_path / "graphiti.kuzu")
        self._driver = KuzuDriver(db=kuzu_db_path)

        # Get LLM client and embedder based on config (LLM-agnostic)
        llm_client = get_graphiti_llm_client()
        embedder = _get_embedder()

        # Initialize Graphiti with custom clients
        self._graphiti = Graphiti(
            graph_driver=self._driver,
            llm_client=llm_client,
            embedder=embedder,
        )

        # Build indices for optimal performance
        try:
            await self._graphiti.build_indices_and_constraints()
        except Exception as e:
            import logging
            logging.debug(f"Could not build indices (may already exist): {e}")

        self._initialized = True

    async def close(self) -> None:
        """Close the graph store and release resources."""
        if self._graphiti is not None:
            await self._graphiti.close()
            self._graphiti = None
            self._driver = None
            self._initialized = False

    async def add_episode(
        self,
        name: str,
        content: str,
        source_type: str = "text",
        source_description: str = "",
        group_id: str = "default",
        reference_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Add an episode to extract entities and relationships.

        Graphiti automatically extracts entities and relationships from the content
        using the configured LLM (Ollama, OpenAI, or Anthropic).

        Args:
            name: Episode identifier/name
            content: Text content to process
            source_type: Type of source ("text" or "json")
            source_description: Description of the data source
            group_id: Group ID for scoping searches
            reference_time: When this information became valid

        Returns:
            Dict with episode info, extracted entities, and relationships
        """
        await self._ensure_initialized()

        from graphiti_core.nodes import EpisodeType

        episode_type = EpisodeType.json if source_type == "json" else EpisodeType.text

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        result = await self._graphiti.add_episode(
            name=name,
            episode_body=content,
            source=episode_type,
            source_description=source_description,
            reference_time=reference_time,
            group_id=group_id,
        )

        return {
            "episode_uuid": result.episode.uuid,
            "entities_count": len(result.nodes),
            "relationships_count": len(result.edges),
            "entities": [
                {"uuid": n.uuid, "name": n.name, "labels": getattr(n, "labels", [])}
                for n in result.nodes
            ],
            "relationships": [
                {"uuid": e.uuid, "fact": e.fact}
                for e in result.edges
            ],
        }

    async def add_entity(self, entity: Entity) -> str:
        """Add a single entity to the graph.

        Note: For bulk entity extraction, use add_episode() which leverages
        the configured LLM for extraction. This method is for manual entity addition.

        Args:
            entity: Entity to add

        Returns:
            Entity UUID from Graphiti
        """
        await self._ensure_initialized()

        # Create an episode with structured data representing the entity
        entity_data = {
            "entity_id": entity.id,
            "name": entity.name,
            "type": entity.entity_type.value,
            "description": entity.description or "",
            "properties": entity.properties,
        }

        result = await self._graphiti.add_episode(
            name=f"entity_{entity.id}",
            episode_body=json.dumps(entity_data),
            source="json",
            source_description=f"Manual entity: {entity.entity_type.value}",
            reference_time=entity.valid_from,
            group_id="entities",
        )

        return result.episode.uuid

    async def add_entities(self, entities: list[Entity]) -> int:
        """Add multiple entities to the graph.

        Args:
            entities: List of entities to add

        Returns:
            Number of entities added
        """
        count = 0
        for entity in entities:
            await self.add_entity(entity)
            count += 1
        return count

    async def search(
        self,
        query: str,
        group_ids: list[str] | None = None,
        num_results: int = 10,
        entity_types: list[EntityType] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the knowledge graph using hybrid semantic + keyword search.

        Args:
            query: Search query
            group_ids: Optional list of group IDs to scope the search
            num_results: Maximum number of results
            entity_types: Optional filter by entity types

        Returns:
            List of matching facts/relationships with metadata
        """
        await self._ensure_initialized()

        search_kwargs: dict[str, Any] = {
            "query": query,
            "num_results": num_results,
        }

        if group_ids:
            search_kwargs["group_ids"] = group_ids

        # Apply entity type filter if specified
        if entity_types:
            from graphiti_core.search.search_filters import SearchFilters
            entity_labels = [et.value for et in entity_types]
            search_kwargs["search_filter"] = SearchFilters(entity_labels=entity_labels)

        results = await self._graphiti.search(**search_kwargs)

        return [
            {
                "uuid": edge.uuid,
                "fact": edge.fact,
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
                "created_at": edge.created_at.isoformat() if edge.created_at else None,
                "episodes": [ep for ep in edge.episodes] if hasattr(edge, "episodes") else [],
            }
            for edge in results
        ]

    async def search_entities(
        self,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Search for entities matching the query.

        Args:
            query: Search query
            group_ids: Optional group IDs to scope search
            limit: Maximum results

        Returns:
            List of matching Entity objects
        """
        # Search returns edges/facts, extract unique entities
        results = await self.search(query, group_ids=group_ids, num_results=limit * 2)

        # Convert to Entity objects (simplified - Graphiti returns edges not nodes directly)
        entities: list[Entity] = []
        seen_uuids: set[str] = set()

        for r in results:
            for node_uuid in [r.get("source_node_uuid"), r.get("target_node_uuid")]:
                if node_uuid and node_uuid not in seen_uuids:
                    seen_uuids.add(node_uuid)
                    # Create a basic entity from the fact
                    entities.append(
                        Entity(
                            id=node_uuid,
                            name=r.get("fact", "")[:100],  # Use fact as name placeholder
                            entity_type=EntityType.CONCEPT,
                            description=r.get("fact"),
                        )
                    )
                    if len(entities) >= limit:
                        return entities

        return entities

    async def get_related(
        self,
        entity_uuid: str,
        relationship_types: list[RelationshipType] | None = None,
        depth: int = 1,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get entities related to a given entity.

        Uses Graphiti's graph distance reranking to find related entities.

        Args:
            entity_uuid: UUID of the center entity
            relationship_types: Optional filter by relationship types
            depth: Traversal depth (not directly supported, uses reranking)
            limit: Maximum results

        Returns:
            List of related facts/edges
        """
        await self._ensure_initialized()

        # Use center_node_uuid for graph-distance based retrieval
        results = await self._graphiti.search(
            query="*",  # Wildcard to get all connected
            center_node_uuid=entity_uuid,
            num_results=limit,
        )

        return [
            {
                "uuid": edge.uuid,
                "fact": edge.fact,
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
            }
            for edge in results
        ]

    async def delete_episode(self, episode_uuid: str) -> bool:
        """Delete an episode and its associated data.

        Args:
            episode_uuid: UUID of the episode to delete

        Returns:
            True if deleted successfully
        """
        await self._ensure_initialized()

        # Graphiti doesn't have direct episode deletion in the public API
        raise NotImplementedError(
            "Episode deletion not yet supported. "
            "Use clear() to reset the entire graph."
        )

    async def clear(self) -> None:
        """Clear all data from the graph store."""
        await self.close()

        import shutil
        if self._db_path.exists():
            shutil.rmtree(self._db_path)
        self._db_path.mkdir(parents=True, exist_ok=True)

    async def count(self) -> dict[str, int]:
        """Get counts of entities and relationships.

        Returns:
            Dict with entity_count and relationship_count
        """
        await self._ensure_initialized()

        # Graphiti doesn't expose direct count methods
        return {
            "entity_count": -1,  # Not directly available
            "relationship_count": -1,
        }

    def __enter__(self) -> GraphStore:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close the store."""
        asyncio.get_event_loop().run_until_complete(self.close())

    async def __aenter__(self) -> GraphStore:
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


def get_graph_store(db_path: Path | str | None = None) -> GraphStore:
    """Factory function to get a GraphStore instance.

    Args:
        db_path: Optional database path override

    Returns:
        GraphStore instance
    """
    return GraphStore(db_path=db_path)
