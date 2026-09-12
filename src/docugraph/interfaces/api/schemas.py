"""Pydantic schemas for the REST API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Search Schemas
# =============================================================================


class SearchRequest(BaseModel):
    """Request body for search endpoint."""

    query: str = Field(..., description="Search query text", min_length=1)
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=100)
    filter: dict[str, Any] | None = Field(default=None, description="Optional filter criteria")


class ChunkResponse(BaseModel):
    """A document chunk in search results."""

    id: str
    document_id: str
    content: str
    start_char: int
    end_char: int
    metadata: dict[str, Any]
    created_at: datetime


class SearchResultResponse(BaseModel):
    """A single search result."""

    chunk: ChunkResponse
    score: float
    highlights: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response body for search endpoint."""

    query: str
    results: list[SearchResultResponse]
    total: int
    took_ms: float


# =============================================================================
# Crawl Schemas
# =============================================================================


class CrawlRequest(BaseModel):
    """Request body for crawl endpoint."""

    url: str = Field(..., description="URL to crawl")
    max_pages: int = Field(default=1, description="Maximum pages to crawl", ge=1, le=1000)
    pattern: str | None = Field(default=None, description="URL pattern filter (regex)")


class CrawlResponse(BaseModel):
    """Response body for crawl endpoint."""

    url: str
    pages_crawled: int
    chunks_indexed: int
    total_chunks: int


# =============================================================================
# Index Schemas
# =============================================================================


class IndexLocalRequest(BaseModel):
    """Request body for indexing local files."""

    path: str = Field(..., description="Path to directory to index")
    recursive: bool = Field(default=True, description="Recursively index subdirectories")
    patterns: list[str] | None = Field(
        default=None, description="File patterns to include (e.g., ['*.md', '*.rst'])"
    )


class IndexGitRequest(BaseModel):
    """Request body for indexing a git repository."""

    url: str = Field(..., description="Git repository URL")
    name: str | None = Field(default=None, description="Custom name for the repository")
    ref: str | None = Field(default=None, description="Branch, tag, or commit to checkout")
    shallow: bool = Field(default=False, description="Shallow clone (depth=1)")
    include_code: bool = Field(default=False, description="Include code files")


class IndexResponse(BaseModel):
    """Response body for index endpoints."""

    source: str
    files_indexed: int
    chunks_indexed: int
    total_chunks: int


class RepoInfo(BaseModel):
    """Information about a cloned repository."""

    name: str
    url: str
    branch: str | None
    commit: str | None
    local_path: str


# =============================================================================
# Memory Schemas
# =============================================================================


class MemorySetRequest(BaseModel):
    """Request body for setting a memory entry."""

    key: str = Field(..., description="Memory key", min_length=1)
    value: Any = Field(..., description="Value to store (any JSON type)")
    repository: str = Field(default="default", description="Repository scope")
    branch: str = Field(default="main", description="Branch scope")


class MemoryGetResponse(BaseModel):
    """Response body for getting a memory entry."""

    key: str
    value: Any
    repository: str
    branch: str
    created_at: datetime | None = None
    accessed_at: datetime | None = None


class MemoryListResponse(BaseModel):
    """Response body for listing memory keys."""

    keys: list[str]
    repository: str
    branch: str
    total: int


class MemoryDeleteResponse(BaseModel):
    """Response body for deleting a memory entry."""

    key: str
    deleted: bool


# =============================================================================
# Graph Schemas
# =============================================================================


class GraphAddRequest(BaseModel):
    """Request body for adding content to the knowledge graph."""

    content: str = Field(..., description="Content to extract entities from")
    name: str | None = Field(default=None, description="Episode name")
    group_id: str = Field(default="default", description="Group ID for scoping")
    source_type: str = Field(default="text", description="Content type (text/json)")


class GraphAddResponse(BaseModel):
    """Response body for adding to the knowledge graph."""

    episode_uuid: str
    entities_count: int
    relationships_count: int
    entities: list[dict[str, Any]]


class GraphSearchRequest(BaseModel):
    """Request body for searching the knowledge graph."""

    query: str = Field(..., description="Search query")
    group_ids: list[str] | None = Field(default=None, description="Group IDs to filter by")
    num_results: int = Field(default=10, description="Maximum results", ge=1, le=100)


class GraphSearchResponse(BaseModel):
    """Response body for knowledge graph search."""

    query: str
    results: list[dict[str, Any]]
    total: int


# =============================================================================
# Stats & Health Schemas
# =============================================================================


class StatsResponse(BaseModel):
    """Response body for stats endpoint."""

    total_chunks: int
    data_directory: str
    embedding_model: str
    embedding_provider: str


class HealthResponse(BaseModel):
    """Response body for health endpoint."""

    status: str
    version: str


# =============================================================================
# Hybrid Search Schemas
# =============================================================================


class HybridSearchRequest(BaseModel):
    """Request body for hybrid search."""

    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, description="Number of results")
    mode: str = Field(default="hybrid", description="Search mode: vector, keyword, hybrid, all")
    fusion: str = Field(default="rrf", description="Fusion strategy: rrf, weighted, interleaved")
    include_graph: bool = Field(default=False, description="Include graph search")


class HybridSearchResultResponse(BaseModel):
    """A single hybrid search result."""

    chunk: ChunkResponse
    score: float
    sources: list[str]
    vector_score: float | None = None
    keyword_score: float | None = None
    graph_score: float | None = None


class HybridSearchResponse(BaseModel):
    """Response body for hybrid search."""

    query: str
    results: list[HybridSearchResultResponse]
    total: int
    took_ms: float
