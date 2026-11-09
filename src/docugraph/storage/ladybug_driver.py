"""A Graphiti driver for LadybugDB.

LadybugDB (PyPI ``ladybug``) exposes the same client API as the archived Kùzu
0.11.3 — ``Database``, ``Connection``, ``AsyncConnection``, the same Cypher
dialect, and the same schema requirements. Graphiti 0.30.2's Kùzu operation
classes therefore run unchanged; this module supplies only the driver object
they are attached to.

Two things differ from graphiti's ``KuzuDriver``, and both matter:

* **Binding.** Omitting a key that the statement references raises on Ladybug,
  while Kùzu bound it as NULL. Graphiti's driver filters ``None`` out of the
  parameters, so this driver passes every supplied key through — ``None``
  becomes a bound NULL. Keys Kùzu's driver drops because the server rejects
  them (``database_``, ``routing_``) are still dropped here.
* **Full-text search.** The FTS extension is database-scoped, so it is loaded
  once on a setup connection *before* the async connection pool is built; every
  pooled connection then inherits it. Loading it after the pool (or per query)
  would leave pool connections unable to run ``QUERY_FTS_INDEX``.

The driver reports ``GraphProvider.KUZU`` on purpose: graphiti's Kùzu-specific
queries and code paths are exactly what Ladybug needs, and graphiti imports
nothing from this module by provider name.

``SCHEMA_QUERIES`` is copied from graphiti-core 0.30.2
(``graphiti_core/driver/kuzu_driver.py``), Copyright 2024 Zep Software, Inc.,
Apache-2.0. See ``NOTICE``.
"""

from __future__ import annotations

import logging
from collections.abc import Coroutine
from typing import Any, Self

import ladybug
from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.kuzu.operations.community_edge_ops import KuzuCommunityEdgeOperations
from graphiti_core.driver.kuzu.operations.community_node_ops import KuzuCommunityNodeOperations
from graphiti_core.driver.kuzu.operations.entity_edge_ops import KuzuEntityEdgeOperations
from graphiti_core.driver.kuzu.operations.entity_node_ops import KuzuEntityNodeOperations
from graphiti_core.driver.kuzu.operations.episode_node_ops import KuzuEpisodeNodeOperations
from graphiti_core.driver.kuzu.operations.episodic_edge_ops import KuzuEpisodicEdgeOperations
from graphiti_core.driver.kuzu.operations.graph_ops import KuzuGraphMaintenanceOperations
from graphiti_core.driver.kuzu.operations.has_episode_edge_ops import KuzuHasEpisodeEdgeOperations
from graphiti_core.driver.kuzu.operations.next_episode_edge_ops import (
    KuzuNextEpisodeEdgeOperations,
)
from graphiti_core.driver.kuzu.operations.saga_node_ops import KuzuSagaNodeOperations
from graphiti_core.driver.kuzu.operations.search_ops import KuzuSearchOperations
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.operations.search_ops import SearchOperations

logger = logging.getLogger(__name__)

# Ladybug requires an explicit schema.
# As Ladybug (like Kùzu) does not support creating full text indexes on edge
# properties, (n:Entity)-[:RELATES_TO]->(m:Entity) is represented as
# (n)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m).
SCHEMA_QUERIES = """
    CREATE NODE TABLE IF NOT EXISTS Episodic (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        created_at TIMESTAMP,
        source STRING,
        source_description STRING,
        content STRING,
        valid_at TIMESTAMP,
        entity_edges STRING[]
    );
    CREATE NODE TABLE IF NOT EXISTS Entity (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        labels STRING[],
        created_at TIMESTAMP,
        name_embedding FLOAT[],
        summary STRING,
        attributes STRING
    );
    CREATE NODE TABLE IF NOT EXISTS Community (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        created_at TIMESTAMP,
        name_embedding FLOAT[],
        summary STRING
    );
    CREATE NODE TABLE IF NOT EXISTS RelatesToNode_ (
        uuid STRING PRIMARY KEY,
        group_id STRING,
        created_at TIMESTAMP,
        name STRING,
        fact STRING,
        fact_embedding FLOAT[],
        episodes STRING[],
        expired_at TIMESTAMP,
        valid_at TIMESTAMP,
        invalid_at TIMESTAMP,
        reference_time TIMESTAMP,
        attributes STRING
    );
    CREATE REL TABLE IF NOT EXISTS RELATES_TO(
        FROM Entity TO RelatesToNode_,
        FROM RelatesToNode_ TO Entity
    );
    CREATE REL TABLE IF NOT EXISTS MENTIONS(
        FROM Episodic TO Entity,
        uuid STRING PRIMARY KEY,
        group_id STRING,
        created_at TIMESTAMP
    );
    CREATE REL TABLE IF NOT EXISTS HAS_MEMBER(
        FROM Community TO Entity,
        FROM Community TO Community,
        uuid STRING,
        group_id STRING,
        created_at TIMESTAMP
    );
    CREATE NODE TABLE IF NOT EXISTS Saga (
        uuid STRING PRIMARY KEY,
        name STRING,
        group_id STRING,
        created_at TIMESTAMP
    );
    CREATE REL TABLE IF NOT EXISTS HAS_EPISODE(
        FROM Saga TO Episodic,
        uuid STRING,
        group_id STRING,
        created_at TIMESTAMP
    );
    CREATE REL TABLE IF NOT EXISTS NEXT_EPISODE(
        FROM Episodic TO Episodic,
        uuid STRING,
        group_id STRING,
        created_at TIMESTAMP
    );
"""


def load_extension_statement(path: str) -> str:
    """The one ``LOAD`` statement, built in one place.

    A path containing a single quote cannot be escaped in a Cypher string
    literal, so it is refused rather than producing a broken statement.
    """
    if "'" in path:
        raise ValueError(f"the extension path contains a single quote: {path!r}")
    return f"LOAD EXTENSION '{path}'"


class LadybugDriver(GraphDriver):
    provider: GraphProvider = GraphProvider.KUZU
    aoss_client: None = None

    def __init__(
        self,
        db: str = ":memory:",
        max_concurrent_queries: int = 1,
        fts_extension: str | None = None,
    ):
        super().__init__()
        self.db = ladybug.Database(db)

        # Database-scoped: one load here serves every connection opened later.
        # This must stay ahead of the AsyncConnection construction below.
        if fts_extension is not None:
            self._load_fts(fts_extension)

        self.setup_schema()

        self.client = ladybug.AsyncConnection(
            self.db, max_concurrent_queries=max_concurrent_queries
        )

        self._entity_node_ops = KuzuEntityNodeOperations()
        self._episode_node_ops = KuzuEpisodeNodeOperations()
        self._community_node_ops = KuzuCommunityNodeOperations()
        self._saga_node_ops = KuzuSagaNodeOperations()
        self._entity_edge_ops = KuzuEntityEdgeOperations()
        self._episodic_edge_ops = KuzuEpisodicEdgeOperations()
        self._community_edge_ops = KuzuCommunityEdgeOperations()
        self._has_episode_edge_ops = KuzuHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = KuzuNextEpisodeEdgeOperations()
        self._search_ops = KuzuSearchOperations()
        self._graph_ops = KuzuGraphMaintenanceOperations()

    def _load_fts(self, path: str) -> None:
        conn = ladybug.Connection(self.db)
        try:
            conn.execute(load_extension_statement(path))
        finally:
            conn.close()

    # --- Operations properties ---

    @property
    def entity_node_ops(self) -> EntityNodeOperations:
        return self._entity_node_ops

    @property
    def episode_node_ops(self) -> EpisodeNodeOperations:
        return self._episode_node_ops

    @property
    def community_node_ops(self) -> CommunityNodeOperations:
        return self._community_node_ops

    @property
    def saga_node_ops(self) -> SagaNodeOperations:
        return self._saga_node_ops

    @property
    def entity_edge_ops(self) -> EntityEdgeOperations:
        return self._entity_edge_ops

    @property
    def episodic_edge_ops(self) -> EpisodicEdgeOperations:
        return self._episodic_edge_ops

    @property
    def community_edge_ops(self) -> CommunityEdgeOperations:
        return self._community_edge_ops

    @property
    def has_episode_edge_ops(self) -> HasEpisodeEdgeOperations:
        return self._has_episode_edge_ops

    @property
    def next_episode_edge_ops(self) -> NextEpisodeEdgeOperations:
        return self._next_episode_edge_ops

    @property
    def search_ops(self) -> SearchOperations:
        return self._search_ops

    @property
    def graph_ops(self) -> GraphMaintenanceOperations:
        return self._graph_ops

    async def execute_query(
        self, cypher_query_: str, **kwargs: Any
    ) -> tuple[list[dict[str, Any]] | list[list[dict[str, Any]]], None, None]:
        # Every supplied key is passed, ``None`` included: Ladybug raises when a
        # referenced parameter is absent, and binds NULL when it is present.
        params = dict(kwargs)
        # Ladybug, like Kùzu, has no per-query database or routing target.
        params.pop("database_", None)
        params.pop("routing_", None)

        try:
            results = await self.client.execute(cypher_query_, parameters=params)
        except Exception as e:
            logged = {k: (v[:5] if isinstance(v, list) else v) for k, v in params.items()}
            logger.error(f"Error executing Ladybug query: {e}\n{cypher_query_}\n{logged}")
            raise

        if not results:
            return [], None, None

        if isinstance(results, list):
            dict_results = [list(result.rows_as_dict()) for result in results]
        else:
            # Ladybug's stub types rows_as_dict() as an iterator of lists or
            # dicts; at runtime a single-result query yields dicts.
            dict_results = list(results.rows_as_dict())  # type: ignore[arg-type]
        return dict_results, None, None  # type: ignore

    def session(self, _database: str | None = None) -> GraphDriverSession:
        return LadybugDriverSession(self)

    async def close(self) -> None:
        # graphiti's KuzuDriver relies on GC. Closing the async connection here
        # releases its worker threads and the file handle deterministically,
        # which is what lets clear() and tests remove the database file.
        if getattr(self, "client", None) is not None:
            self.client.close()
            self.client = None  # type: ignore[assignment]

    def delete_all_indexes(
        self,
        database_: str = "",  # noqa: ARG002 -- graphiti's Kùzu call shape
    ) -> Coroutine[Any, Any, None]:
        # graphiti's base class types this as a sync method returning a
        # coroutine, and only the Neo4j path awaits it. Both engines create
        # their indexes in the schema, so there is nothing to drop.
        async def _noop() -> None:
            return None

        return _noop()

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        # Ladybug has no dynamic index creation; the schema is created in
        # setup_schema(). Required by the abstract base class.
        pass

    def setup_schema(self) -> None:
        conn = ladybug.Connection(self.db)
        try:
            conn.execute(SCHEMA_QUERIES)
        finally:
            conn.close()


class LadybugDriverSession(GraphDriverSession):
    provider = GraphProvider.KUZU

    def __init__(self, driver: LadybugDriver):
        self.driver = driver

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        pass

    async def close(self) -> None:
        # The driver's connection is reused; the session owns nothing.
        pass

    async def execute_write(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list[tuple[str, dict[str, Any]]], **kwargs: Any) -> Any:
        if isinstance(query, list):
            for cypher, params in query:
                await self.driver.execute_query(cypher, **params)
        else:
            await self.driver.execute_query(query, **kwargs)
        return None
