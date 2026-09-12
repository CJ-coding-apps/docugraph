"""Ingestion pipeline: crawling, file processing, and chunking."""

from docugraph.ingestion.api_docs import (
    APIDocConfig,
    APIDocParser,
    APIDocType,
    APIEndpoint,
    APIParameter,
    APISchema,
    GraphQLType,
    ParsedAPISpec,
    get_api_doc_parser,
)
from docugraph.ingestion.chunker import (
    Chunker,
    ChunkerConfig,
    ChunkingStrategy,
    get_chunker,
)
from docugraph.ingestion.crawler import DocCrawler
from docugraph.ingestion.git_indexer import (
    GitIndexer,
    GitIndexerConfig,
    RepoInfo,
    get_git_indexer,
)
from docugraph.ingestion.local_files import (
    FileType,
    IndexedFile,
    LocalFileConfig,
    LocalFileIndexer,
    get_local_file_indexer,
)

__all__ = [
    # Crawler
    "DocCrawler",
    # Chunker
    "Chunker",
    "ChunkerConfig",
    "ChunkingStrategy",
    "get_chunker",
    # Local Files
    "FileType",
    "IndexedFile",
    "LocalFileConfig",
    "LocalFileIndexer",
    "get_local_file_indexer",
    # Git Indexer
    "GitIndexer",
    "GitIndexerConfig",
    "RepoInfo",
    "get_git_indexer",
    # API Docs
    "APIDocType",
    "APIParameter",
    "APIEndpoint",
    "APISchema",
    "GraphQLType",
    "ParsedAPISpec",
    "APIDocConfig",
    "APIDocParser",
    "get_api_doc_parser",
]
