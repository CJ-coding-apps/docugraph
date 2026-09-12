"""Unit tests for the chunker module."""

import pytest

from docugraph.core.models import Document
from docugraph.ingestion.chunker import (
    Chunker,
    ChunkerConfig,
    ChunkingStrategy,
    get_chunker,
)


class TestChunkerConfig:
    """Tests for ChunkerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkerConfig()
        assert config.strategy == ChunkingStrategy.SEMANTIC
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkerConfig(
            strategy=ChunkingStrategy.FIXED,
            chunk_size=1000,
            chunk_overlap=100,
        )
        assert config.strategy == ChunkingStrategy.FIXED
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 100


class TestChunker:
    """Tests for Chunker class."""

    def test_chunker_creation(self):
        """Test chunker instantiation."""
        chunker = Chunker()
        assert chunker is not None

    def test_chunker_with_config(self):
        """Test chunker with custom config."""
        config = ChunkerConfig(chunk_size=256)
        chunker = Chunker(config=config)
        assert chunker._config.chunk_size == 256

    def test_chunk_document_basic(self):
        """Test basic document chunking."""
        chunker = Chunker()
        doc = Document(
            content="This is a test document. " * 50,
            content_type="text",
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 0
        assert all(c.document_id == doc.id for c in chunks)

    def test_chunk_document_short(self):
        """Test chunking a short document."""
        # Use a config with lower min_chunk_size to accept short content
        config = ChunkerConfig(min_chunk_size=10)
        chunker = Chunker(config=config)
        doc = Document(
            content="Short content.",
            content_type="text",
        )
        chunks = chunker.chunk_document(doc)
        # Short docs should still produce at least one chunk
        assert len(chunks) >= 1

    def test_chunk_document_empty(self):
        """Test chunking an empty document."""
        chunker = Chunker()
        doc = Document(
            content="",
            content_type="text",
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 0

    def test_chunk_preserves_metadata(self):
        """Test that chunks preserve document metadata."""
        chunker = Chunker()
        doc = Document(
            content="Content " * 100,
            content_type="markdown",
            source_url="https://example.com",
            metadata={"title": "Test Doc"},
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 0
        # Check metadata is preserved
        for chunk in chunks:
            assert "source_url" in chunk.metadata or chunk.metadata.get("title")

    def test_fixed_strategy(self):
        """Test fixed-size chunking strategy."""
        config = ChunkerConfig(
            strategy=ChunkingStrategy.FIXED,
            chunk_size=200,
            chunk_overlap=50,
            min_chunk_size=50,  # Lower threshold to ensure chunks are created
        )
        chunker = Chunker(config=config)
        doc = Document(
            content="Word " * 200,  # ~1000 chars
            content_type="text",
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_semantic_strategy(self):
        """Test semantic chunking strategy."""
        config = ChunkerConfig(strategy=ChunkingStrategy.SEMANTIC)
        chunker = Chunker(config=config)
        doc = Document(
            content="""# Section 1

This is the first section with some content.

# Section 2

This is the second section with more content.

## Subsection 2.1

Additional details here.
""",
            content_type="markdown",
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 0

    def test_sentence_strategy(self):
        """Test sentence-based chunking strategy."""
        config = ChunkerConfig(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=200,
        )
        chunker = Chunker(config=config)
        doc = Document(
            content="This is sentence one. This is sentence two. This is sentence three. " * 20,
            content_type="text",
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 0

    def test_chunk_positions(self):
        """Test that chunk positions are correct."""
        config = ChunkerConfig(
            strategy=ChunkingStrategy.FIXED,
            chunk_size=50,
            chunk_overlap=10,
        )
        chunker = Chunker(config=config)
        content = "A" * 200
        doc = Document(content=content, content_type="text")
        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char <= len(content)
            assert chunk.start_char < chunk.end_char


class TestGetChunker:
    """Tests for get_chunker factory function."""

    def test_get_chunker_default(self):
        """Test factory with default config."""
        chunker = get_chunker()
        assert isinstance(chunker, Chunker)

    def test_get_chunker_with_config(self):
        """Test factory with custom config."""
        config = ChunkerConfig(chunk_size=1024)
        chunker = get_chunker(config=config)
        assert chunker._config.chunk_size == 1024
