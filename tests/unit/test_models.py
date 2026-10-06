"""Unit tests for core data models.

The last class is a sweep rather than a per-model test: it discovers every
pydantic model in the package and holds all their default timestamps to the
same rule, so a new model is covered the day it is written.
"""

import importlib
import inspect
import re
from datetime import datetime
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel, Field

from docugraph.agents.coding_agent import AgentMessage
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


# --------------------------------------------------------------------------
# Default timestamps
# --------------------------------------------------------------------------

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
_MODEL_CLASS = re.compile(r"^class\s+\w+\([^)]*BaseModel[^)]*\):", re.MULTILINE)


def _model_classes() -> list[type[BaseModel]]:
    """Every BaseModel subclass defined in the package, found from the source.

    Found by scanning rather than by listing, so adding a model cannot quietly
    exempt it from the check below.
    """
    found: list[type[BaseModel]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if not _MODEL_CLASS.search(path.read_text(encoding="utf-8")):
            continue
        module_name = ".".join(path.relative_to(SRC_ROOT).with_suffix("").parts)
        module = importlib.import_module(module_name)
        found.extend(
            obj
            for _name, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, BaseModel) and obj is not BaseModel and obj.__module__ == module_name
        )
    return found


def _mentions_datetime(annotation: Any) -> bool:
    """True for `datetime` and for optional spellings of it (`datetime | None`)."""
    return annotation is datetime or datetime in get_args(annotation)


def _timestamp_defaults(model: type[BaseModel]) -> list[tuple[str, datetime]]:
    """Every datetime field of ``model`` that has a default value, not a hole."""
    defaults: list[tuple[str, datetime]] = []
    for name, info in model.model_fields.items():
        if not _mentions_datetime(info.annotation):
            continue
        if info.default_factory is not None:
            value: Any = info.default_factory()
        elif isinstance(info.default, datetime):
            value = info.default
        else:
            continue  # required, or defaulted to None -- nothing to check
        if isinstance(value, datetime):
            defaults.append((name, value))
    return defaults


def _naive_timestamp_fields(model: type[BaseModel]) -> list[str]:
    """The fields of ``model`` whose default timestamp carries no timezone."""
    return [name for name, value in _timestamp_defaults(model) if value.utcoffset() is None]


class TestDefaultTimestamps:
    """Every default timestamp is timezone-aware, and every model is checked.

    Two reasons this is its own test rather than a line in each model's:

    It is not cosmetic. A naive default is later compared against aware values
    -- the crawler's cache TTL is one place -- and aware-minus-naive raises
    rather than answering, so the failure surfaces far from the field.

    And it is the one hole the deprecation rule cannot cover. A warning raised
    inside a `default_factory` is attributed to the frame that called it, which
    is pydantic, so `error::DeprecationWarning:docugraph` never sees it. This
    test is what makes that gap closed rather than widened.
    """

    def test_the_sweep_finds_models_to_check(self) -> None:
        """If discovery breaks, every assertion below becomes vacuous."""
        assert _model_classes(), "no models were discovered, so none were checked"

    def test_every_model_default_timestamp_is_timezone_aware(self) -> None:
        naive: list[str] = []
        checked = 0
        for model in _model_classes():
            for name, value in _timestamp_defaults(model):
                checked += 1
                if value.utcoffset() is None:
                    naive.append(f"{model.__name__}.{name}")

        assert naive == [], f"naive default timestamps: {', '.join(naive)}"

        # Eight today: Document 2, Chunk 1, Entity 2, Relationship 1,
        # MemoryEntry 2. Exact rather than a floor, so that losing a model to
        # broken discovery fails here too. Bump it when a model gains one.
        assert checked == 8, f"expected 8 default timestamps, checked {checked}"

    def test_the_check_notices_a_naive_default(self) -> None:
        """The control for the sweep: a deliberately naive default is reported."""

        class _Sloppy(BaseModel):
            created_at: datetime = Field(default_factory=datetime.now)  # naive on purpose

        assert _naive_timestamp_fields(_Sloppy) == ["created_at"]

    def test_the_agent_message_timestamp_is_aware_too(self) -> None:
        """The one default timestamp outside pydantic (agents/coding_agent.py).

        A dataclass rather than a model, so the sweep above does not reach it --
        but the defect it guards against is identical.
        """
        message = AgentMessage(role="user", content="hello")

        assert message.timestamp.utcoffset() is not None
