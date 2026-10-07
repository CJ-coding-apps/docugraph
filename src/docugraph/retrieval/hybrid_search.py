"""Hybrid search combining vector, keyword, and graph retrieval.

Implements multiple fusion strategies:
- Reciprocal Rank Fusion (RRF) - default, robust
- Weighted Score Fusion - for fine-tuned weights
- Interleaved - alternating results from each source
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from docugraph.core.models import Chunk, SearchResult


class FusionStrategy(StrEnum):
    """Strategy for combining search results."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted"  # Weighted score combination
    INTERLEAVED = "interleaved"  # Alternating results


class SearchMode(StrEnum):
    """Search modes available."""

    VECTOR = "vector"  # Vector similarity only
    KEYWORD = "keyword"  # BM25/keyword only
    HYBRID = "hybrid"  # Vector + keyword
    GRAPH = "graph"  # Graph traversal only
    ALL = "all"  # Vector + keyword + graph


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    # Weights for each search type (used in weighted fusion)
    vector_weight: float = 0.5
    keyword_weight: float = 0.3
    graph_weight: float = 0.2

    # RRF parameter (higher = more emphasis on top ranks)
    rrf_k: int = 60

    # Whether to include graph search
    include_graph: bool = False

    # Minimum score threshold (0-1)
    min_score: float = 0.0

    # Deduplication
    deduplicate: bool = True


@dataclass(frozen=True)
class GraphLegStatus:
    """What happened to the graph leg of the most recent search.

    The graph leg is optional -- it needs a graph store on disk, which needs an
    LLM. When it cannot run, the caller still gets vector and/or keyword
    results, and that is the whole point of degrading. What must not happen is
    the *reason* disappearing: "asked for graph results and got none" would then
    be indistinguishable from "the graph had nothing to say", and a silent
    failure would read as a successful empty answer.
    """

    requested: bool = False
    available: bool = False
    reason: str | None = None


@dataclass
class HybridResult:
    """Result from hybrid search with source attribution."""

    chunk: Chunk
    score: float
    sources: list[str] = field(default_factory=list)  # Which searches found this
    vector_score: float | None = None
    keyword_score: float | None = None
    graph_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_search_result(self) -> SearchResult:
        """Convert to standard SearchResult."""
        return SearchResult(chunk=self.chunk, score=self.score)


class HybridRetriever:
    """Hybrid retriever combining multiple search strategies.

    Combines results from:
    - Vector search (semantic similarity via embeddings)
    - Keyword search (BM25 full-text search)
    - Graph search (knowledge graph traversal, optional)

    Uses configurable fusion strategies to combine results.
    """

    def __init__(
        self,
        vector_store: Any | None = None,
        graph_store: Any | None = None,
        config: HybridSearchConfig | None = None,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            vector_store: VectorStore instance (created if not provided)
            graph_store: GraphStore instance (optional, for graph search)
            config: Search configuration
        """
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._config = config or HybridSearchConfig()
        # Why the graph store could not be built or queried during this call.
        # Carried forward to `graph_status`; see GraphLegStatus.
        self._graph_store_error: str | None = None
        # Set by every `search()` call: whether the graph leg was asked for and
        # whether it actually ran.
        self.graph_status = GraphLegStatus()

    def _get_vector_store(self) -> Any:
        """Lazy load vector store."""
        if self._vector_store is None:
            from docugraph.storage.vector_store import VectorStore

            self._vector_store = VectorStore()
        return self._vector_store

    def _get_graph_store(self) -> Any | None:
        """Lazy load graph store if configured.

        A failure here is not raised -- the graph is optional -- but the reason
        is kept in ``self._graph_store_error`` rather than discarded, so the
        caller can be told the graph did not run.
        """
        if self._graph_store is None and self._config.include_graph:
            try:
                from docugraph.storage.graph_store import GraphStore

                self._graph_store = GraphStore()
            except Exception as e:  # nosec B110 -- graph is optional; degrade to vector/keyword
                self._graph_store_error = f"the graph store could not be opened: {e}"
        return self._graph_store

    async def asearch(
        self,
        query: str,
        top_k: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        fusion: FusionStrategy = FusionStrategy.RRF,
        filters: dict[str, Any] | None = None,
        config: HybridSearchConfig | None = None,
    ) -> list[HybridResult]:
        """Perform hybrid search.

        This is the implementation. Callers that are already async -- the MCP
        server, the agent loop -- await it directly, which is the only correct
        thing to do from inside a running event loop.

        Args:
            query: Search query text
            top_k: Number of results to return
            mode: Search mode (vector, keyword, hybrid, graph, all)
            fusion: Fusion strategy for combining results
            filters: Optional filters (passed to vector store)
            config: Override default config for this search

        Returns:
            List of hybrid results with scores and source attribution
        """
        cfg = config or self._config

        # Collect results from each source
        results_by_source: dict[str, list[tuple[str, float]]] = {}

        # Vector search
        if mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL):
            vector_results = self._vector_search(query, top_k * 2, filters)
            results_by_source["vector"] = vector_results

        # Keyword search
        if mode in (SearchMode.KEYWORD, SearchMode.HYBRID, SearchMode.ALL):
            keyword_results = self._keyword_search(query, top_k * 2, filters)
            results_by_source["keyword"] = keyword_results

        # Graph search. Optional: it is skipped or fails without stopping the
        # search, but what happened is recorded either way and reported through
        # `graph_status`. It is awaited in place -- `asearch` is already async,
        # so there is no loop to acquire here.
        graph_requested = mode in (SearchMode.GRAPH, SearchMode.ALL)
        self._graph_store_error = None
        graph_available = False
        graph_reason: str | None = None

        if graph_requested and cfg.include_graph:
            try:
                graph_results = await self._graph_search(query, top_k * 2)
                results_by_source["graph"] = graph_results
                graph_available = True
            except Exception as e:  # nosec B110 -- graph search may fail if not configured
                graph_reason = f"the graph search could not be run: {e}"

            # A store that failed to open reports through `_graph_store_error`
            # rather than as an exception, so it is resolved after the fact.
            if self._graph_store_error is not None:
                graph_available = False
                graph_reason = self._graph_store_error
        elif graph_requested:
            graph_reason = "graph search is disabled (set include_graph=true)"

        self.graph_status = GraphLegStatus(
            requested=graph_requested,
            available=graph_available,
            reason=graph_reason,
        )

        # Fuse results
        if fusion == FusionStrategy.RRF:
            fused = self._fuse_rrf(results_by_source, cfg.rrf_k)
        elif fusion == FusionStrategy.WEIGHTED:
            fused = self._fuse_weighted(results_by_source, cfg)
        else:  # INTERLEAVED
            fused = self._fuse_interleaved(results_by_source)

        # Deduplicate if configured
        if cfg.deduplicate:
            fused = self._deduplicate(fused)

        # Apply minimum score filter (fused tuples are (chunk_id, score, sources))
        if cfg.min_score > 0:
            fused = [r for r in fused if r[1] >= cfg.min_score]

        # Retrieve full chunks and build results
        return self._build_results(fused, results_by_source, top_k)

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        fusion: FusionStrategy = FusionStrategy.RRF,
        filters: dict[str, Any] | None = None,
        config: HybridSearchConfig | None = None,
    ) -> list[HybridResult]:
        """Perform hybrid search from synchronous code.

        A thin wrapper over ``asearch``, for callers that have no event loop --
        the CLI. It must not be called from inside a running loop: this drives
        a fresh loop, and calling it from within one raises. Async callers
        await ``asearch`` instead; they do not use this.

        Args:
            As ``asearch``.

        Returns:
            List of hybrid results with scores and source attribution
        """
        return asyncio.run(
            self.asearch(
                query,
                top_k=top_k,
                mode=mode,
                fusion=fusion,
                filters=filters,
                config=config,
            )
        )

    def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[tuple[str, float]]:
        """Perform vector similarity search.

        Returns:
            List of (chunk_id, score) tuples
        """
        store = self._get_vector_store()
        filter_expr = self._build_filter_expr(filters) if filters else None

        results = store.search(query, top_k=top_k, filter_expr=filter_expr)
        return [(r.chunk.id, r.score) for r in results]

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[tuple[str, float]]:
        """Perform keyword/BM25 search.

        Returns:
            List of (chunk_id, score) tuples
        """
        store = self._get_vector_store()
        filter_expr = self._build_filter_expr(filters) if filters else None

        # Keyword-side source: LanceDB native hybrid (vector+FTS) with any
        # filters applied, so keyword mode respects the same scoping as vector.
        results = store.search_hybrid(query, top_k=top_k, filter_expr=filter_expr)
        return [(r.chunk.id, r.score) for r in results]

    async def _graph_search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Perform graph-based search.

        Returns:
            List of (chunk_id, score) tuples
        """
        store = self._get_graph_store()
        if store is None:
            return []

        try:
            results = await store.search(query, num_results=top_k)
            # Graph search returns facts, not chunks directly
            # We'll use the fact UUID as an identifier and assign scores
            return [
                (r.get("uuid", ""), 1.0 / (i + 1))  # Score by rank
                for i, r in enumerate(results)
            ]
        except Exception as e:
            self._graph_store_error = f"the graph store could not be queried: {e}"
            return []

    def _fuse_rrf(
        self,
        results_by_source: dict[str, list[tuple[str, float]]],
        k: int = 60,
    ) -> list[tuple[str, float, list[str]]]:
        """Fuse results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each source

        Args:
            results_by_source: Dict mapping source name to (id, score) tuples
            k: RRF constant (default 60)

        Returns:
            List of (id, fused_score, sources) tuples sorted by score
        """
        scores: dict[str, float] = {}
        sources: dict[str, list[str]] = {}

        for source, results in results_by_source.items():
            for rank, (doc_id, _) in enumerate(results, 1):
                rrf_score = 1.0 / (k + rank)
                scores[doc_id] = scores.get(doc_id, 0) + rrf_score

                if doc_id not in sources:
                    sources[doc_id] = []
                sources[doc_id].append(source)

        # Sort by score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, sources.get(doc_id, [])) for doc_id, score in sorted_results]

    def _fuse_weighted(
        self,
        results_by_source: dict[str, list[tuple[str, float]]],
        config: HybridSearchConfig,
    ) -> list[tuple[str, float, list[str]]]:
        """Fuse results using weighted score combination.

        Args:
            results_by_source: Dict mapping source name to (id, score) tuples
            config: Config with weight settings

        Returns:
            List of (id, fused_score, sources) tuples sorted by score
        """
        weights = {
            "vector": config.vector_weight,
            "keyword": config.keyword_weight,
            "graph": config.graph_weight,
        }

        scores: dict[str, float] = {}
        sources: dict[str, list[str]] = {}

        for source, results in results_by_source.items():
            weight = weights.get(source, 0.0)
            for doc_id, score in results:
                weighted_score = score * weight
                scores[doc_id] = scores.get(doc_id, 0) + weighted_score

                if doc_id not in sources:
                    sources[doc_id] = []
                sources[doc_id].append(source)

        # Sort by score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, sources.get(doc_id, [])) for doc_id, score in sorted_results]

    def _fuse_interleaved(
        self,
        results_by_source: dict[str, list[tuple[str, float]]],
    ) -> list[tuple[str, float, list[str]]]:
        """Fuse results by interleaving from each source.

        Takes results round-robin from each source.

        Returns:
            List of (id, fused_score, sources) tuples
        """
        seen: set[str] = set()
        fused: list[tuple[str, float, list[str]]] = []

        # Get iterators for each source
        iterators = {source: iter(results) for source, results in results_by_source.items()}

        # Round-robin interleave
        rank = 0
        while iterators:
            exhausted = []
            for source, it in iterators.items():
                try:
                    doc_id, score = next(it)
                    if doc_id not in seen:
                        seen.add(doc_id)
                        rank += 1
                        # Assign score based on interleaved rank
                        fused.append((doc_id, 1.0 / rank, [source]))
                except StopIteration:
                    exhausted.append(source)

            for source in exhausted:
                del iterators[source]

        return fused

    def _deduplicate(
        self,
        results: list[tuple[str, float, list[str]]],
    ) -> list[tuple[str, float, list[str]]]:
        """Remove duplicate document IDs, keeping highest score."""
        seen: set[str] = set()
        deduped: list[tuple[str, float, list[str]]] = []

        for doc_id, score, sources in results:
            if doc_id not in seen:
                seen.add(doc_id)
                deduped.append((doc_id, score, sources))

        return deduped

    def _build_filter_expr(self, filters: dict[str, Any]) -> str:
        """Build SQL-like filter expression from filter dict."""
        conditions = []
        for key, value in filters.items():
            if isinstance(value, str):
                conditions.append(f"{key} = '{value}'")
            elif isinstance(value, (int, float)):
                conditions.append(f"{key} = {value}")
            elif isinstance(value, list):
                values = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in value)
                conditions.append(f"{key} IN ({values})")

        return " AND ".join(conditions) if conditions else ""

    def _build_results(
        self,
        fused: list[tuple[str, float, list[str]]],
        results_by_source: dict[str, list[tuple[str, float]]],
        top_k: int,
    ) -> list[HybridResult]:
        """Build HybridResult objects from fused scores.

        Args:
            fused: List of (id, score, sources) tuples
            results_by_source: Original results with per-source scores
            top_k: Maximum results to return

        Returns:
            List of HybridResult objects
        """
        # Build lookup for per-source scores
        source_scores: dict[str, dict[str, float]] = {}
        for source, results in results_by_source.items():
            source_scores[source] = dict(results)

        # Get chunks from vector store
        store = self._get_vector_store()
        hybrid_results: list[HybridResult] = []

        for doc_id, score, sources in fused[:top_k]:
            chunk = store.get_chunk(doc_id)
            if chunk is None:
                continue

            hybrid_results.append(
                HybridResult(
                    chunk=chunk,
                    score=score,
                    sources=sources,
                    vector_score=source_scores.get("vector", {}).get(doc_id),
                    keyword_score=source_scores.get("keyword", {}).get(doc_id),
                    graph_score=source_scores.get("graph", {}).get(doc_id),
                )
            )

        return hybrid_results


def get_hybrid_retriever(
    config: HybridSearchConfig | None = None,
    include_graph: bool = False,
) -> HybridRetriever:
    """Factory function to get a HybridRetriever instance.

    Args:
        config: Optional search configuration
        include_graph: Whether to include graph search

    Returns:
        HybridRetriever instance
    """
    if config is None:
        config = HybridSearchConfig(include_graph=include_graph)
    else:
        config.include_graph = include_graph

    return HybridRetriever(config=config)
