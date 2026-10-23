"""Benchmark arms: each is one retrieval configuration under measurement.

An arm is a callable ``(query_text, top_k) -> list[(chunk_id, score)]``. The
runner treats arms uniformly; everything arm-specific -- which store method,
which fusion, whether a reranker sits on top -- lives here.

The ``lancedb-hybrid`` arm refuses to report numbers when the store says
native hybrid fell back to vector-only: a benchmark that silently scored
vector search under the hybrid label would be measuring a lie, which is the
precise failure this benchmark exists to expose (see
``VectorStore.last_hybrid_fallback_reason``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Arm = Callable[[str, int], list[tuple[str, float]]]

# Candidate depth handed to a reranker. Deeper than top_k, because a reranker
# can only promote what retrieval already surfaced.
RERANK_CANDIDATES = 20


class BenchArmError(RuntimeError):
    """An arm could not run as the configuration it claims to measure."""


def _pairs(results: Any) -> list[tuple[str, float]]:
    return [(r.chunk.id, float(r.score)) for r in results]


def build_arms(
    vector_store: Any,
    *,
    include_rerank: bool = False,
) -> dict[str, Arm]:
    """The arm registry over one ingested store.

    Reranking arms are opt-in: the local cross-encoder weighs ~2.3 GB and has
    its own opt-in workflow, mirroring how tests/unit/test_reranker.py is
    handled.
    """
    from docugraph.retrieval.hybrid_search import (
        FusionStrategy,
        HybridRetriever,
        SearchMode,
    )

    retriever = HybridRetriever(vector_store=vector_store)

    def vector(query: str, top_k: int) -> list[tuple[str, float]]:
        return _pairs(vector_store.search(query, top_k=top_k))

    def fts(query: str, top_k: int) -> list[tuple[str, float]]:
        return _pairs(vector_store.search_fts(query, top_k=top_k))

    def lancedb_hybrid(query: str, top_k: int) -> list[tuple[str, float]]:
        results = vector_store.search_hybrid(query, top_k=top_k)
        reason = getattr(vector_store, "last_hybrid_fallback_reason", None)
        if reason is not None:
            raise BenchArmError(
                f"lancedb-hybrid fell back to vector-only ({reason}); "
                "refusing to score vector search under the hybrid label"
            )
        return _pairs(results)

    def rrf(query: str, top_k: int) -> list[tuple[str, float]]:
        results = retriever.search(
            query, top_k=top_k, mode=SearchMode.HYBRID, fusion=FusionStrategy.RRF
        )
        return [(r.chunk.id, float(r.score)) for r in results]

    def weighted(query: str, top_k: int) -> list[tuple[str, float]]:
        results = retriever.search(
            query, top_k=top_k, mode=SearchMode.HYBRID, fusion=FusionStrategy.WEIGHTED
        )
        return [(r.chunk.id, float(r.score)) for r in results]

    arms: dict[str, Arm] = {
        "vector": vector,
        "fts": fts,
        "lancedb-hybrid": lancedb_hybrid,
        "rrf": rrf,
        "weighted": weighted,
    }

    if include_rerank:
        from docugraph.core.models import SearchResult
        from docugraph.retrieval.reranker import FastembedReranker

        reranker = FastembedReranker()

        def _reranked(base: Arm) -> Arm:
            def arm(query: str, top_k: int) -> list[tuple[str, float]]:
                candidates = base(query, max(top_k, RERANK_CANDIDATES))
                results = []
                for chunk_id, score in candidates:
                    chunk = vector_store.get_chunk(chunk_id)
                    if chunk is not None:
                        results.append(SearchResult(chunk=chunk, score=score))
                reranked = reranker.rerank(query=query, results=results, top_k=top_k)
                return [(r.chunk.id, float(r.final_score)) for r in reranked]

            return arm

        arms["vector+rerank"] = _reranked(vector)
        arms["rrf+rerank"] = _reranked(rrf)

    return arms
