"""Retrieval system: hybrid search, reranking, and context building."""

from docugraph.retrieval.context_builder import (
    BuiltContext,
    ContextBuilder,
    ContextConfig,
    ContextFormat,
    ContextSource,
    get_context_builder,
)
from docugraph.retrieval.hybrid_search import (
    FusionStrategy,
    HybridResult,
    HybridRetriever,
    HybridSearchConfig,
    SearchMode,
    get_hybrid_retriever,
)
from docugraph.retrieval.reranker import (
    CohereReranker,
    FastembedReranker,
    LLMReranker,
    PassthroughReranker,
    RerankerBase,
    RerankerType,
    RerankResult,
    get_reranker,
)

__all__ = [
    # Hybrid Search
    "FusionStrategy",
    "SearchMode",
    "HybridSearchConfig",
    "HybridResult",
    "HybridRetriever",
    "get_hybrid_retriever",
    # Reranking
    "RerankerType",
    "RerankResult",
    "RerankerBase",
    "FastembedReranker",
    "CohereReranker",
    "LLMReranker",
    "PassthroughReranker",
    "get_reranker",
    # Context Building
    "ContextFormat",
    "ContextConfig",
    "ContextSource",
    "BuiltContext",
    "ContextBuilder",
    "get_context_builder",
]
