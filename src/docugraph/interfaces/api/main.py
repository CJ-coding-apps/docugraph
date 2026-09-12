"""FastAPI REST API for DocuGraph AI."""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from docugraph import __version__
from docugraph.core.config import get_config
from docugraph.interfaces.api.schemas import (
    ChunkResponse,
    CrawlRequest,
    CrawlResponse,
    GraphAddRequest,
    GraphAddResponse,
    GraphSearchRequest,
    GraphSearchResponse,
    HealthResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResultResponse,
    IndexGitRequest,
    IndexLocalRequest,
    IndexResponse,
    MemoryDeleteResponse,
    MemoryGetResponse,
    MemoryListResponse,
    MemorySetRequest,
    RepoInfo,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
    StatsResponse,
)
from docugraph.storage.vector_store import VectorStore

# Global instances
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get or create the vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: initialize vector store
    global _vector_store
    _vector_store = VectorStore()
    yield
    # Shutdown: cleanup if needed
    pass


# Create FastAPI app
app = FastAPI(
    title="DocuGraph AI API",
    description="REST API for intelligent documentation RAG",
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Stats
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version=__version__)


@app.get("/v1/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Get statistics about indexed content."""
    config = get_config()
    vector_store = get_vector_store()

    return StatsResponse(
        total_chunks=vector_store.count(),
        data_directory=str(config.storage.data_dir),
        embedding_model=config.embeddings.model,
        embedding_provider=config.embeddings.provider.value,
    )


# =============================================================================
# Search Endpoints
# =============================================================================


@app.post("/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Search indexed documentation."""
    start_time = time.time()

    vector_store = get_vector_store()

    try:
        results = vector_store.search(request.query, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # Convert to response format
    result_responses = []
    for result in results:
        chunk = result.chunk
        result_responses.append(
            SearchResultResponse(
                chunk=ChunkResponse(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata=chunk.metadata,
                    created_at=chunk.created_at,
                ),
                score=result.score,
                highlights=result.highlights,
            )
        )

    elapsed_ms = (time.time() - start_time) * 1000

    return SearchResponse(
        query=request.query,
        results=result_responses,
        total=len(result_responses),
        took_ms=elapsed_ms,
    )


@app.post("/v1/search/hybrid", response_model=HybridSearchResponse)
async def hybrid_search(request: HybridSearchRequest) -> HybridSearchResponse:
    """Perform hybrid search combining vector, keyword, and graph search."""
    start_time = time.time()

    from docugraph.retrieval.hybrid_search import (
        FusionStrategy,
        HybridRetriever,
        HybridSearchConfig,
        SearchMode,
    )

    # Map request params to enums
    mode_map = {
        "vector": SearchMode.VECTOR,
        "keyword": SearchMode.KEYWORD,
        "hybrid": SearchMode.HYBRID,
        "graph": SearchMode.GRAPH,
        "all": SearchMode.ALL,
    }
    fusion_map = {
        "rrf": FusionStrategy.RRF,
        "weighted": FusionStrategy.WEIGHTED,
        "interleaved": FusionStrategy.INTERLEAVED,
    }

    mode = mode_map.get(request.mode.lower(), SearchMode.HYBRID)
    fusion = fusion_map.get(request.fusion.lower(), FusionStrategy.RRF)

    config = HybridSearchConfig(include_graph=request.include_graph)
    retriever = HybridRetriever(config=config)

    try:
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            mode=mode,
            fusion=fusion,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")

    # Convert to response
    result_responses = []
    for result in results:
        chunk = result.chunk
        result_responses.append(
            HybridSearchResultResponse(
                chunk=ChunkResponse(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata=chunk.metadata,
                    created_at=chunk.created_at,
                ),
                score=result.score,
                sources=result.sources,
                vector_score=result.vector_score,
                keyword_score=result.keyword_score,
                graph_score=result.graph_score,
            )
        )

    elapsed_ms = (time.time() - start_time) * 1000

    return HybridSearchResponse(
        query=request.query,
        results=result_responses,
        total=len(result_responses),
        took_ms=elapsed_ms,
    )


# =============================================================================
# Crawl & Index Endpoints
# =============================================================================


@app.post("/v1/crawl", response_model=CrawlResponse)
async def crawl(request: CrawlRequest) -> CrawlResponse:
    """Crawl a URL and index its content."""
    from docugraph.ingestion.chunker import Chunker
    from docugraph.ingestion.crawler import DocCrawler

    crawler = DocCrawler()
    chunker = Chunker()
    vector_store = get_vector_store()

    try:
        # Crawl
        if request.max_pages == 1:
            documents = [await crawler.crawl_single(request.url)]
        else:
            documents = await crawler.crawl_site(
                request.url,
                max_pages=request.max_pages,
                url_pattern=request.pattern,
            )

        # Chunk and index
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        count = vector_store.add_chunks(all_chunks)

        return CrawlResponse(
            url=request.url,
            pages_crawled=len(documents),
            chunks_indexed=count,
            total_chunks=vector_store.count(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")


@app.post("/v1/index/local", response_model=IndexResponse)
async def index_local(request: IndexLocalRequest) -> IndexResponse:
    """Index local documentation files."""
    from docugraph.ingestion.chunker import Chunker
    from docugraph.ingestion.local_files import LocalFileConfig, LocalFileIndexer

    path = Path(request.path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {request.path}")

    config = LocalFileConfig()
    if request.patterns:
        config.include_patterns = request.patterns

    indexer = LocalFileIndexer(config=config)
    chunker = Chunker()
    vector_store = get_vector_store()

    try:
        documents = indexer.index_to_documents(path, recursive=request.recursive)

        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        count = vector_store.add_chunks(all_chunks)

        return IndexResponse(
            source=request.path,
            files_indexed=len(documents),
            chunks_indexed=count,
            total_chunks=vector_store.count(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index failed: {str(e)}")


@app.post("/v1/index/git", response_model=IndexResponse)
async def index_git(request: IndexGitRequest) -> IndexResponse:
    """Clone and index a git repository."""
    from docugraph.ingestion.chunker import Chunker
    from docugraph.ingestion.git_indexer import GitIndexer, GitIndexerConfig

    config = GitIndexerConfig(
        clone_depth=1 if request.shallow else None,
        ref=request.ref,
        include_code=request.include_code,
    )

    indexer = GitIndexer(config=config)
    chunker = Chunker()
    vector_store = get_vector_store()

    try:
        repo_info, documents = indexer.clone_and_index(
            request.url,
            name=request.name,
            ref=request.ref,
        )

        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        count = vector_store.add_chunks(all_chunks)

        return IndexResponse(
            source=f"{repo_info.name} @ {repo_info.branch or 'HEAD'}",
            files_indexed=len(documents),
            chunks_indexed=count,
            total_chunks=vector_store.count(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Git index failed: {str(e)}")


@app.get("/v1/repos", response_model=list[RepoInfo])
async def list_repos() -> list[RepoInfo]:
    """List all cloned repositories."""
    from docugraph.ingestion.git_indexer import GitIndexer

    indexer = GitIndexer()
    repos = indexer.list_repos()

    return [
        RepoInfo(
            name=r.name,
            url=r.url,
            branch=r.branch,
            commit=r.commit_hash[:12] if r.commit_hash else None,
            local_path=str(r.local_path),
        )
        for r in repos
    ]


@app.delete("/v1/index")
async def clear_index() -> dict[str, str]:
    """Clear all indexed content."""
    vector_store = get_vector_store()
    vector_store.clear()
    return {"status": "cleared"}


# =============================================================================
# Memory Endpoints
# =============================================================================


@app.post("/v1/memory", response_model=MemoryGetResponse)
async def memory_set(request: MemorySetRequest) -> MemoryGetResponse:
    """Store a value in agent memory."""
    from docugraph.storage.memory_store import MemoryStore

    store = MemoryStore()
    entry = store.set(
        key=request.key,
        value=request.value,
        repository=request.repository,
        branch=request.branch,
    )

    return MemoryGetResponse(
        key=entry.key,
        value=entry.value,
        repository=entry.repository,
        branch=entry.branch,
        created_at=entry.created_at,
        accessed_at=entry.accessed_at,
    )


@app.get("/v1/memory/{key}", response_model=MemoryGetResponse)
async def memory_get(
    key: str,
    repository: str = "default",
    branch: str = "main",
) -> MemoryGetResponse:
    """Retrieve a value from agent memory."""
    from docugraph.storage.memory_store import MemoryStore

    store = MemoryStore()
    value = store.get(key, repository=repository, branch=branch)

    if value is None:
        raise HTTPException(status_code=404, detail=f"Key not found: {key}")

    return MemoryGetResponse(
        key=key,
        value=value,
        repository=repository,
        branch=branch,
    )


@app.get("/v1/memory", response_model=MemoryListResponse)
async def memory_list(
    repository: str = "default",
    branch: str = "main",
) -> MemoryListResponse:
    """List all memory keys for a repository/branch."""
    from docugraph.storage.memory_store import MemoryStore

    store = MemoryStore()
    keys = store.list_keys(repository=repository, branch=branch)

    return MemoryListResponse(
        keys=keys,
        repository=repository,
        branch=branch,
        total=len(keys),
    )


@app.delete("/v1/memory/{key}", response_model=MemoryDeleteResponse)
async def memory_delete(
    key: str,
    repository: str = "default",
    branch: str = "main",
) -> MemoryDeleteResponse:
    """Delete a memory entry."""
    from docugraph.storage.memory_store import MemoryStore

    store = MemoryStore()
    deleted = store.delete(key, repository=repository, branch=branch)

    return MemoryDeleteResponse(key=key, deleted=deleted)


# =============================================================================
# Graph Endpoints
# =============================================================================


@app.post("/v1/graph/add", response_model=GraphAddResponse)
async def graph_add(request: GraphAddRequest) -> GraphAddResponse:
    """Add content to the knowledge graph."""
    from docugraph.storage.graph_store import GraphStore

    store = GraphStore()

    try:
        episode_name = request.name or f"episode_{hash(request.content) % 10000}"
        result = await store.add_episode(
            name=episode_name,
            content=request.content,
            source_type=request.source_type,
            group_id=request.group_id,
        )

        return GraphAddResponse(
            episode_uuid=result["episode_uuid"],
            entities_count=result["entities_count"],
            relationships_count=result["relationships_count"],
            entities=result["entities"],
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Graph store requires LLM. {str(e)}",
        )
    finally:
        await store.close()


@app.post("/v1/graph/search", response_model=GraphSearchResponse)
async def graph_search(request: GraphSearchRequest) -> GraphSearchResponse:
    """Search the knowledge graph."""
    from docugraph.storage.graph_store import GraphStore

    store = GraphStore()

    try:
        results = await store.search(
            query=request.query,
            group_ids=request.group_ids,
            num_results=request.num_results,
        )

        return GraphSearchResponse(
            query=request.query,
            results=results,
            total=len(results),
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Graph store requires LLM. {str(e)}",
        )
    finally:
        await store.close()


@app.delete("/v1/graph")
async def graph_clear() -> dict[str, str]:
    """Clear the knowledge graph."""
    from docugraph.storage.graph_store import GraphStore

    store = GraphStore()
    await store.clear()
    return {"status": "cleared"}


# =============================================================================
# Run
# =============================================================================


def run() -> None:
    """Run the API server."""
    import uvicorn

    config = get_config()
    uvicorn.run(
        "docugraph.interfaces.api.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.debug,
    )


if __name__ == "__main__":
    run()
