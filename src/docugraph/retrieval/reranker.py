"""Reranking module for improving search result relevance.

Supports multiple reranking strategies:
- Cross-encoder (sentence-transformers) - high quality, local
- Cohere Rerank API - cloud-based
- LLM-based reranking - flexible but expensive
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from docugraph.core.models import Chunk, SearchResult


class RerankerType(str, Enum):
    """Available reranker types."""

    CROSS_ENCODER = "cross-encoder"
    COHERE = "cohere"
    LLM = "llm"
    NONE = "none"  # Pass-through, no reranking


@dataclass
class RerankResult:
    """Result from reranking with original and new scores."""

    chunk: Chunk
    original_score: float
    rerank_score: float
    final_score: float  # Combined score
    metadata: dict[str, Any] | None = None


class RerankerBase(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank search results.

        Args:
            query: The search query
            results: Initial search results to rerank
            top_k: Optional limit on returned results

        Returns:
            Reranked results with scores
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the reranker name."""
        ...


class CrossEncoderReranker(RerankerBase):
    """Cross-encoder reranker using sentence-transformers.

    Cross-encoders jointly encode query and document for more accurate
    relevance scoring than bi-encoders (used in initial retrieval).

    Default model: ms-marco-MiniLM-L-6-v2 (fast, good quality)
    Alternative: ms-marco-MiniLM-L-12-v2 (slower, better quality)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
        batch_size: int = 32,
        score_weight: float = 0.7,
    ) -> None:
        """Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
            device: Device to run on (auto, cpu, cuda, mps)
            batch_size: Batch size for inference
            score_weight: Weight for rerank score in final score (0-1)
        """
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._score_weight = score_weight
        self._model: Any = None

    def _get_device(self) -> str:
        """Determine best device."""
        if self._device != "auto":
            return self._device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _ensure_model(self) -> Any:
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            ) from e

        device = self._get_device()
        self._model = CrossEncoder(self._model_name, device=device)
        return self._model

    @property
    def name(self) -> str:
        return f"cross-encoder:{self._model_name}"

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank using cross-encoder.

        Args:
            query: Search query
            results: Results to rerank
            top_k: Optional limit on results

        Returns:
            Reranked results
        """
        if not results:
            return []

        model = self._ensure_model()

        # Prepare query-document pairs
        pairs = [(query, r.chunk.content) for r in results]

        # Get cross-encoder scores
        scores = model.predict(pairs, batch_size=self._batch_size)

        # Normalize scores to 0-1 range (cross-encoder scores can be any range)
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score != min_score else 1.0

        normalized_scores = [(s - min_score) / score_range for s in scores]

        # Combine with original scores
        reranked: list[RerankResult] = []
        for result, rerank_score in zip(results, normalized_scores):
            original_weight = 1.0 - self._score_weight
            final_score = (
                self._score_weight * rerank_score + original_weight * result.score
            )
            reranked.append(
                RerankResult(
                    chunk=result.chunk,
                    original_score=result.score,
                    rerank_score=rerank_score,
                    final_score=final_score,
                )
            )

        # Sort by final score descending
        reranked.sort(key=lambda x: x.final_score, reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked


class CohereReranker(RerankerBase):
    """Cohere Rerank API reranker.

    Uses Cohere's cloud-based reranking API for high-quality results.
    Requires COHERE_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        api_key: str | None = None,
        score_weight: float = 0.8,
    ) -> None:
        """Initialize Cohere reranker.

        Args:
            model: Cohere rerank model name
            api_key: API key (or set COHERE_API_KEY env var)
            score_weight: Weight for rerank score in final score
        """
        import os

        self._model = model
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        self._score_weight = score_weight
        self._client: Any = None

        if not self._api_key:
            raise ValueError(
                "COHERE_API_KEY environment variable required for Cohere reranking"
            )

    def _ensure_client(self) -> Any:
        """Lazy load Cohere client."""
        if self._client is not None:
            return self._client

        try:
            import cohere
        except ImportError as e:
            raise ImportError(
                "cohere package required. Install with: pip install cohere"
            ) from e

        self._client = cohere.Client(self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return f"cohere:{self._model}"

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank using Cohere API.

        Args:
            query: Search query
            results: Results to rerank
            top_k: Optional limit on results

        Returns:
            Reranked results
        """
        if not results:
            return []

        client = self._ensure_client()

        # Prepare documents
        documents = [r.chunk.content for r in results]

        # Call Cohere rerank
        response = client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_k or len(results),
        )

        # Build results map
        result_map = {i: r for i, r in enumerate(results)}

        reranked: list[RerankResult] = []
        for item in response.results:
            original = result_map[item.index]
            original_weight = 1.0 - self._score_weight
            final_score = (
                self._score_weight * item.relevance_score
                + original_weight * original.score
            )
            reranked.append(
                RerankResult(
                    chunk=original.chunk,
                    original_score=original.score,
                    rerank_score=item.relevance_score,
                    final_score=final_score,
                )
            )

        return reranked


class LLMReranker(RerankerBase):
    """LLM-based reranker using language models for relevance scoring.

    More flexible than cross-encoders but slower and more expensive.
    Uses the configured LLM provider.
    """

    def __init__(
        self,
        score_weight: float = 0.6,
        batch_size: int = 5,
    ) -> None:
        """Initialize LLM reranker.

        Args:
            score_weight: Weight for rerank score in final score
            batch_size: Number of documents to score per LLM call
        """
        self._score_weight = score_weight
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return "llm"

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank using LLM scoring.

        Args:
            query: Search query
            results: Results to rerank
            top_k: Optional limit on results

        Returns:
            Reranked results
        """
        import asyncio

        if not results:
            return []

        # Run async reranking synchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._rerank_async(query, results, top_k)
                    )
                    return future.result()
            return loop.run_until_complete(self._rerank_async(query, results, top_k))
        except RuntimeError:
            return asyncio.run(self._rerank_async(query, results, top_k))

    async def _rerank_async(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None,
    ) -> list[RerankResult]:
        """Async implementation of LLM reranking."""
        from docugraph.core.llm import get_llm_provider

        llm = get_llm_provider()

        # Score each document
        reranked: list[RerankResult] = []

        for i in range(0, len(results), self._batch_size):
            batch = results[i : i + self._batch_size]
            scores = await self._score_batch(llm, query, batch)

            for result, score in zip(batch, scores):
                original_weight = 1.0 - self._score_weight
                final_score = (
                    self._score_weight * score + original_weight * result.score
                )
                reranked.append(
                    RerankResult(
                        chunk=result.chunk,
                        original_score=result.score,
                        rerank_score=score,
                        final_score=final_score,
                    )
                )

        # Sort by final score
        reranked.sort(key=lambda x: x.final_score, reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    async def _score_batch(
        self,
        llm: Any,
        query: str,
        results: list[SearchResult],
    ) -> list[float]:
        """Score a batch of documents using LLM.

        Args:
            llm: LLM provider instance
            query: Search query
            results: Results to score

        Returns:
            List of relevance scores (0-1)
        """
        # Build prompt for relevance scoring
        documents = "\n\n".join(
            f"Document {i+1}:\n{r.chunk.content[:500]}"
            for i, r in enumerate(results)
        )

        prompt = f"""Rate the relevance of each document to the query on a scale of 0-10.
Query: {query}

{documents}

For each document, output only a number 0-10 on a separate line.
Output exactly {len(results)} numbers, one per line."""

        try:
            response = await llm.generate(
                prompt=prompt,
                system_prompt="You are a relevance scoring assistant. Output only numbers.",
                temperature=0.0,
            )

            # Parse scores from response
            lines = response.strip().split("\n")
            scores = []
            for line in lines:
                try:
                    # Extract first number from line
                    num = float("".join(c for c in line if c.isdigit() or c == "."))
                    scores.append(min(num / 10.0, 1.0))  # Normalize to 0-1
                except (ValueError, IndexError):
                    scores.append(0.5)  # Default score

            # Ensure we have enough scores
            while len(scores) < len(results):
                scores.append(0.5)

            return scores[: len(results)]

        except Exception:
            # Fallback: return neutral scores
            return [0.5] * len(results)


class PassthroughReranker(RerankerBase):
    """No-op reranker that passes results through unchanged."""

    @property
    def name(self) -> str:
        return "none"

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Pass through results without reranking."""
        reranked = [
            RerankResult(
                chunk=r.chunk,
                original_score=r.score,
                rerank_score=r.score,
                final_score=r.score,
            )
            for r in results
        ]

        if top_k:
            reranked = reranked[:top_k]

        return reranked


def get_reranker(
    reranker_type: RerankerType = RerankerType.CROSS_ENCODER,
    **kwargs: Any,
) -> RerankerBase:
    """Factory function to get a reranker instance.

    Args:
        reranker_type: Type of reranker to use
        **kwargs: Additional arguments passed to reranker constructor

    Returns:
        Reranker instance
    """
    if reranker_type == RerankerType.CROSS_ENCODER:
        return CrossEncoderReranker(**kwargs)
    elif reranker_type == RerankerType.COHERE:
        return CohereReranker(**kwargs)
    elif reranker_type == RerankerType.LLM:
        return LLMReranker(**kwargs)
    else:
        return PassthroughReranker()
