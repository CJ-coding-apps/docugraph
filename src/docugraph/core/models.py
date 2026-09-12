"""Core data models for DocuGraph AI."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid4())


class ContentType(StrEnum):
    """Type of document content."""

    MARKDOWN = "markdown"
    CODE = "code"
    API_SPEC = "api_spec"
    TEXT = "text"


class EntityType(StrEnum):
    """Type of knowledge graph entity."""

    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    PACKAGE = "package"
    ENDPOINT = "endpoint"
    PARAMETER = "parameter"
    CONCEPT = "concept"
    PATTERN = "pattern"
    DOCUMENTATION = "documentation"


class RelationshipType(StrEnum):
    """Type of relationship between entities."""

    CALLS = "CALLS"
    IMPORTS = "IMPORTS"
    INHERITS = "INHERITS"
    IMPLEMENTS = "IMPLEMENTS"
    DEPENDS_ON = "DEPENDS_ON"
    USED_BY = "USED_BY"
    DESCRIBES = "DESCRIBES"
    EXAMPLE_OF = "EXAMPLE_OF"
    RELATED_TO = "RELATED_TO"
    SUPERSEDES = "SUPERSEDES"
    DEPRECATES = "DEPRECATES"


class Document(BaseModel):
    """A crawled or indexed document."""

    id: str = Field(default_factory=generate_id)
    source_url: str | None = None
    source_path: str | None = None
    title: str | None = None
    content: str
    content_type: ContentType = ContentType.MARKDOWN
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def __hash__(self) -> int:
        return hash(self.id)


class Chunk(BaseModel):
    """A chunk of a document for vector storage."""

    id: str = Field(default_factory=generate_id)
    document_id: str
    content: str
    embedding: list[float] | None = None
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def __hash__(self) -> int:
        return hash(self.id)


class Entity(BaseModel):
    """A knowledge graph entity (node)."""

    id: str = Field(default_factory=generate_id)
    name: str
    entity_type: EntityType
    description: str | None = None
    source_chunks: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    valid_from: datetime = Field(default_factory=datetime.utcnow)
    valid_to: datetime | None = None

    def __hash__(self) -> int:
        return hash(self.id)


class Relationship(BaseModel):
    """A knowledge graph relationship (edge)."""

    id: str = Field(default_factory=generate_id)
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    properties: dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def __hash__(self) -> int:
        return hash(self.id)


class MemoryEntry(BaseModel):
    """An entry in the agent's memory store."""

    id: str = Field(default_factory=generate_id)
    repository: str
    branch: str = "main"
    key: str
    value: Any
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)

    def __hash__(self) -> int:
        return hash(self.id)


class SearchResult(BaseModel):
    """A search result with relevance score."""

    chunk: Chunk
    score: float
    source_document: Document | None = None
    highlights: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
