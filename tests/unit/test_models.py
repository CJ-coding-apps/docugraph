"""Unit tests for core data models."""

from docugraph.core.models import (
    Chunk,
    Document,
    Entity,
    EntityType,
    MemoryEntry,
    Relationship,
    RelationshipType,
)


class TestDocument:
    """Tests for Document model."""

    def test_document_creation(self):
        """Test basic document creation."""
        doc = Document(
            content="Test content",
            content_type="markdown",
        )
        assert doc.content == "Test content"
        assert doc.content_type == "markdown"
        assert doc.id is not None
        assert doc.created_at is not None

    def test_document_with_metadata(self):
        """Test document with metadata."""
        doc = Document(
            content="Test",
            content_type="text",
            source_url="https://example.com",
            metadata={"key": "value"},
        )
        assert doc.source_url == "https://example.com"
        assert doc.metadata["key"] == "value"

    def test_document_with_source_path(self):
        """Test document with local file source."""
        doc = Document(
            content="Local file content",
            content_type="markdown",
            source_path="/path/to/file.md",
        )
        assert doc.source_path == "/path/to/file.md"


class TestChunk:
    """Tests for Chunk model."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            document_id="doc-123",
            content="Chunk content",
            start_char=0,
            end_char=13,
        )
        assert chunk.document_id == "doc-123"
        assert chunk.content == "Chunk content"
        assert chunk.start_char == 0
        assert chunk.end_char == 13

    def test_chunk_with_metadata(self):
        """Test chunk with metadata."""
        chunk = Chunk(
            document_id="doc-123",
            content="Test",
            start_char=0,
            end_char=4,
            metadata={"section": "intro"},
        )
        assert chunk.metadata["section"] == "intro"


class TestEntity:
    """Tests for Entity model."""

    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = Entity(
            name="MyClass",
            entity_type=EntityType.CLASS,
        )
        assert entity.name == "MyClass"
        assert entity.entity_type == EntityType.CLASS
        assert entity.id is not None

    def test_entity_with_description(self):
        """Test entity with description and properties."""
        entity = Entity(
            name="process_data",
            entity_type=EntityType.FUNCTION,
            description="Processes input data",
            properties={"returns": "dict"},
        )
        assert entity.description == "Processes input data"
        assert entity.properties["returns"] == "dict"

    def test_entity_types(self):
        """Test all entity types."""
        types = [
            EntityType.FUNCTION,
            EntityType.CLASS,
            EntityType.MODULE,
            EntityType.PACKAGE,
            EntityType.ENDPOINT,
            EntityType.PARAMETER,
            EntityType.CONCEPT,
            EntityType.PATTERN,
            EntityType.DOCUMENTATION,
        ]
        for et in types:
            entity = Entity(name="test", entity_type=et)
            assert entity.entity_type == et


class TestRelationship:
    """Tests for Relationship model."""

    def test_relationship_creation(self):
        """Test basic relationship creation."""
        rel = Relationship(
            source_id="entity-1",
            target_id="entity-2",
            relationship_type=RelationshipType.CALLS,
        )
        assert rel.source_id == "entity-1"
        assert rel.target_id == "entity-2"
        assert rel.relationship_type == RelationshipType.CALLS

    def test_relationship_types(self):
        """Test all relationship types."""
        types = [
            RelationshipType.CALLS,
            RelationshipType.IMPORTS,
            RelationshipType.INHERITS,
            RelationshipType.IMPLEMENTS,
            RelationshipType.DEPENDS_ON,
            RelationshipType.USED_BY,
            RelationshipType.DESCRIBES,
            RelationshipType.EXAMPLE_OF,
            RelationshipType.RELATED_TO,
            RelationshipType.SUPERSEDES,
            RelationshipType.DEPRECATES,
        ]
        for rt in types:
            rel = Relationship(
                source_id="a",
                target_id="b",
                relationship_type=rt,
            )
            assert rel.relationship_type == rt


class TestMemoryEntry:
    """Tests for MemoryEntry model."""

    def test_memory_entry_creation(self):
        """Test basic memory entry creation."""
        entry = MemoryEntry(
            key="test_key",
            value="test_value",
            repository="default",
        )
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.repository == "default"
        assert entry.branch == "main"

    def test_memory_entry_with_scope(self):
        """Test memory entry with custom scope."""
        entry = MemoryEntry(
            key="config",
            value={"setting": True},
            repository="my-repo",
            branch="feature",
        )
        assert entry.repository == "my-repo"
        assert entry.branch == "feature"
        assert entry.value == {"setting": True}
