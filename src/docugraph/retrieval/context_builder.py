"""Context builder for assembling LLM-ready context from search results.

Handles:
- Token budget management
- Source attribution formatting
- Memory context injection
- Deduplication and prioritization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from docugraph.core.models import Chunk, MemoryEntry, SearchResult
from docugraph.retrieval.hybrid_search import HybridResult
from docugraph.retrieval.reranker import RerankResult


class ContextFormat(str, Enum):
    """Output format for context."""

    MARKDOWN = "markdown"  # Markdown with headers and code blocks
    PLAIN = "plain"  # Plain text with separators
    XML = "xml"  # XML tags for structured parsing
    JSON = "json"  # JSON for programmatic use


@dataclass
class ContextConfig:
    """Configuration for context building."""

    # Token limits
    max_tokens: int = 8000
    reserve_tokens: int = 1000  # Reserve for response
    chunk_token_estimate: float = 1.3  # chars per token estimate

    # Content settings
    include_sources: bool = True
    include_metadata: bool = False
    include_scores: bool = False

    # Deduplication
    deduplicate: bool = True
    similarity_threshold: float = 0.9  # For content dedup

    # Memory injection
    include_memory: bool = True
    memory_position: str = "before"  # before, after, or interleaved

    # Formatting
    format: ContextFormat = ContextFormat.MARKDOWN
    section_separator: str = "\n\n---\n\n"
    max_chunk_chars: int = 2000  # Truncate very long chunks


@dataclass
class ContextSource:
    """A source included in the context."""

    chunk_id: str
    content: str
    source_url: str | None = None
    source_path: str | None = None
    title: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuiltContext:
    """Result of context building."""

    context: str
    sources: list[ContextSource]
    total_tokens: int
    truncated: bool
    memory_included: bool
    source_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context": self.context,
            "sources": [
                {
                    "chunk_id": s.chunk_id,
                    "source_url": s.source_url,
                    "source_path": s.source_path,
                    "title": s.title,
                    "score": s.score,
                }
                for s in self.sources
            ],
            "total_tokens": self.total_tokens,
            "truncated": self.truncated,
            "memory_included": self.memory_included,
            "source_count": self.source_count,
        }


class ContextBuilder:
    """Builds LLM-ready context from search results.

    Combines search results with optional memory context,
    manages token budgets, and formats for LLM consumption.
    """

    def __init__(self, config: ContextConfig | None = None) -> None:
        """Initialize context builder.

        Args:
            config: Context building configuration
        """
        self._config = config or ContextConfig()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple character-based estimate. For production,
        consider using tiktoken or similar.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return int(len(text) / self._config.chunk_token_estimate)

    def _truncate_chunk(self, content: str) -> str:
        """Truncate chunk content if too long.

        Args:
            content: Chunk content

        Returns:
            Truncated content with indicator if needed
        """
        max_chars = self._config.max_chunk_chars
        if len(content) <= max_chars:
            return content
        return content[:max_chars] + "\n... [truncated]"

    def _normalize_results(
        self,
        results: list[SearchResult | HybridResult | RerankResult | Chunk],
    ) -> list[ContextSource]:
        """Normalize various result types to ContextSource.

        Args:
            results: Mixed result types from search

        Returns:
            List of normalized ContextSource objects
        """
        sources: list[ContextSource] = []

        for result in results:
            if isinstance(result, SearchResult):
                chunk = result.chunk
                score = result.score
            elif isinstance(result, HybridResult):
                chunk = result.chunk
                score = result.score
            elif isinstance(result, RerankResult):
                chunk = result.chunk
                score = result.final_score
            elif isinstance(result, Chunk):
                chunk = result
                score = None
            else:
                continue

            sources.append(
                ContextSource(
                    chunk_id=chunk.id,
                    content=self._truncate_chunk(chunk.content),
                    source_url=chunk.metadata.get("source_url"),
                    source_path=chunk.metadata.get("source_path"),
                    title=chunk.metadata.get("title"),
                    score=score,
                    metadata=chunk.metadata,
                )
            )

        return sources

    def _deduplicate_sources(
        self,
        sources: list[ContextSource],
    ) -> list[ContextSource]:
        """Remove duplicate or near-duplicate sources.

        Uses simple content similarity for deduplication.

        Args:
            sources: Sources to deduplicate

        Returns:
            Deduplicated sources
        """
        if not self._config.deduplicate:
            return sources

        seen_content: set[str] = set()
        deduped: list[ContextSource] = []

        for source in sources:
            # Create a fingerprint from first N chars
            fingerprint = source.content[:200].lower().strip()
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                deduped.append(source)

        return deduped

    def _format_source_markdown(self, source: ContextSource, index: int) -> str:
        """Format a source as Markdown.

        Args:
            source: Source to format
            index: 1-based index for citation

        Returns:
            Markdown formatted string
        """
        lines: list[str] = []

        # Header with source info
        title = source.title or f"Source {index}"
        lines.append(f"### [{index}] {title}")

        if self._config.include_sources:
            if source.source_url:
                lines.append(f"*Source: {source.source_url}*")
            elif source.source_path:
                lines.append(f"*File: {source.source_path}*")

        if self._config.include_scores and source.score is not None:
            lines.append(f"*Relevance: {source.score:.3f}*")

        lines.append("")  # Blank line before content
        lines.append(source.content)

        return "\n".join(lines)

    def _format_source_plain(self, source: ContextSource, index: int) -> str:
        """Format a source as plain text.

        Args:
            source: Source to format
            index: 1-based index for citation

        Returns:
            Plain text formatted string
        """
        lines: list[str] = []

        title = source.title or f"Source {index}"
        lines.append(f"[{index}] {title}")

        if self._config.include_sources:
            if source.source_url:
                lines.append(f"Source: {source.source_url}")
            elif source.source_path:
                lines.append(f"File: {source.source_path}")

        lines.append("")
        lines.append(source.content)

        return "\n".join(lines)

    def _format_source_xml(self, source: ContextSource, index: int) -> str:
        """Format a source as XML.

        Args:
            source: Source to format
            index: 1-based index for citation

        Returns:
            XML formatted string
        """
        import html

        lines: list[str] = []
        lines.append(f'<source index="{index}">')

        title = source.title or f"Source {index}"
        lines.append(f"  <title>{html.escape(title)}</title>")

        if source.source_url:
            lines.append(f"  <url>{html.escape(source.source_url)}</url>")
        if source.source_path:
            lines.append(f"  <path>{html.escape(source.source_path)}</path>")

        if self._config.include_scores and source.score is not None:
            lines.append(f"  <score>{source.score:.3f}</score>")

        lines.append(f"  <content>{html.escape(source.content)}</content>")
        lines.append("</source>")

        return "\n".join(lines)

    def _format_source(self, source: ContextSource, index: int) -> str:
        """Format a source according to configured format.

        Args:
            source: Source to format
            index: 1-based index for citation

        Returns:
            Formatted string
        """
        if self._config.format == ContextFormat.MARKDOWN:
            return self._format_source_markdown(source, index)
        elif self._config.format == ContextFormat.PLAIN:
            return self._format_source_plain(source, index)
        elif self._config.format == ContextFormat.XML:
            return self._format_source_xml(source, index)
        elif self._config.format == ContextFormat.JSON:
            import json

            return json.dumps(
                {
                    "index": index,
                    "title": source.title,
                    "url": source.source_url,
                    "path": source.source_path,
                    "score": source.score,
                    "content": source.content,
                }
            )
        else:
            return self._format_source_plain(source, index)

    def _format_memory(self, memories: list[MemoryEntry]) -> str:
        """Format memory entries for context.

        Args:
            memories: Memory entries to format

        Returns:
            Formatted memory section
        """
        if not memories:
            return ""

        if self._config.format == ContextFormat.MARKDOWN:
            lines = ["## Session Context\n"]
            for mem in memories:
                lines.append(f"- **{mem.key}**: {mem.value}")
            return "\n".join(lines)

        elif self._config.format == ContextFormat.XML:
            lines = ["<session_context>"]
            for mem in memories:
                lines.append(f'  <memory key="{mem.key}">{mem.value}</memory>')
            lines.append("</session_context>")
            return "\n".join(lines)

        else:  # PLAIN or JSON
            lines = ["Session Context:"]
            for mem in memories:
                lines.append(f"  {mem.key}: {mem.value}")
            return "\n".join(lines)

    def build(
        self,
        results: list[SearchResult | HybridResult | RerankResult | Chunk],
        query: str | None = None,
        memories: list[MemoryEntry] | None = None,
        config: ContextConfig | None = None,
    ) -> BuiltContext:
        """Build context from search results.

        Args:
            results: Search results to include
            query: Optional query for context
            memories: Optional memory entries to include
            config: Override default config for this build

        Returns:
            Built context with metadata
        """
        cfg = config or self._config
        available_tokens = cfg.max_tokens - cfg.reserve_tokens

        # Normalize and deduplicate
        sources = self._normalize_results(results)
        sources = self._deduplicate_sources(sources)

        # Build memory section if configured
        memory_text = ""
        memory_included = False
        if cfg.include_memory and memories:
            memory_text = self._format_memory(memories)
            memory_included = True
            # Account for memory tokens
            available_tokens -= self._estimate_tokens(memory_text)

        # Build source sections within token budget
        formatted_sources: list[str] = []
        included_sources: list[ContextSource] = []
        current_tokens = 0
        truncated = False

        for i, source in enumerate(sources, 1):
            formatted = self._format_source(source, i)
            source_tokens = self._estimate_tokens(formatted)

            if current_tokens + source_tokens > available_tokens:
                truncated = True
                break

            formatted_sources.append(formatted)
            included_sources.append(source)
            current_tokens += source_tokens

        # Assemble final context
        separator = cfg.section_separator
        context_parts: list[str] = []

        # Add query context if provided
        if query:
            if cfg.format == ContextFormat.MARKDOWN:
                context_parts.append(f"## Query\n{query}")
            elif cfg.format == ContextFormat.XML:
                context_parts.append(f"<query>{query}</query>")
            else:
                context_parts.append(f"Query: {query}")

        # Add memory before or after sources
        if memory_text and cfg.memory_position == "before":
            context_parts.append(memory_text)

        # Add document header
        if cfg.format == ContextFormat.MARKDOWN:
            context_parts.append("## Relevant Documentation")
        elif cfg.format == ContextFormat.XML:
            context_parts.append("<documentation>")

        # Add sources
        context_parts.extend(formatted_sources)

        if cfg.format == ContextFormat.XML:
            context_parts.append("</documentation>")

        # Add memory after if configured
        if memory_text and cfg.memory_position == "after":
            context_parts.append(memory_text)

        # Join with appropriate separator
        if cfg.format == ContextFormat.XML:
            context = "\n".join(context_parts)
        else:
            context = separator.join(context_parts)

        total_tokens = self._estimate_tokens(context)

        return BuiltContext(
            context=context,
            sources=included_sources,
            total_tokens=total_tokens,
            truncated=truncated,
            memory_included=memory_included,
            source_count=len(included_sources),
        )

    def build_with_citations(
        self,
        results: list[SearchResult | HybridResult | RerankResult | Chunk],
        query: str | None = None,
        memories: list[MemoryEntry] | None = None,
    ) -> tuple[str, str]:
        """Build context with a separate citations section.

        Returns both the main context and a citations/references section
        that can be appended to LLM responses.

        Args:
            results: Search results to include
            query: Optional query
            memories: Optional memory entries

        Returns:
            Tuple of (context, citations)
        """
        built = self.build(results, query, memories)

        # Build citations section
        citations_lines: list[str] = []
        if self._config.format == ContextFormat.MARKDOWN:
            citations_lines.append("## Sources")
            for i, source in enumerate(built.sources, 1):
                title = source.title or f"Source {i}"
                if source.source_url:
                    citations_lines.append(f"[{i}] [{title}]({source.source_url})")
                elif source.source_path:
                    citations_lines.append(f"[{i}] {title} - `{source.source_path}`")
                else:
                    citations_lines.append(f"[{i}] {title}")
        else:
            citations_lines.append("Sources:")
            for i, source in enumerate(built.sources, 1):
                title = source.title or f"Source {i}"
                loc = source.source_url or source.source_path or "N/A"
                citations_lines.append(f"[{i}] {title} - {loc}")

        citations = "\n".join(citations_lines)

        return built.context, citations


def get_context_builder(config: ContextConfig | None = None) -> ContextBuilder:
    """Factory function to get a ContextBuilder instance.

    Args:
        config: Optional configuration

    Returns:
        ContextBuilder instance
    """
    return ContextBuilder(config=config)
