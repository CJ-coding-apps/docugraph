"""Entity extraction from documents using LLM-based NER.

Supports:
- Code entities (functions, classes, modules)
- Documentation entities (concepts, patterns, APIs)
- Custom entity types via configuration

LLM-agnostic: Works with Ollama (local), OpenAI, or Anthropic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from docugraph.core.llm import get_llm_provider
from docugraph.core.models import Document, Entity, EntityType


class ExtractionStrategy(str, Enum):
    """Entity extraction strategies."""

    LLM = "llm"  # LLM-based extraction (most accurate)
    REGEX = "regex"  # Regex-based extraction (fast, limited)
    HYBRID = "hybrid"  # Combine LLM and regex


@dataclass
class ExtractionConfig:
    """Configuration for entity extraction."""

    strategy: ExtractionStrategy = ExtractionStrategy.HYBRID

    # Entity types to extract
    entity_types: list[EntityType] = field(
        default_factory=lambda: [
            EntityType.FUNCTION,
            EntityType.CLASS,
            EntityType.MODULE,
            EntityType.CONCEPT,
            EntityType.ENDPOINT,
        ]
    )

    # LLM settings
    temperature: float = 0.0
    max_tokens: int = 2000

    # Batch settings
    batch_size: int = 5  # Documents per LLM call

    # Minimum confidence threshold
    min_confidence: float = 0.5


@dataclass
class ExtractedEntity:
    """An extracted entity with metadata."""

    name: str
    entity_type: EntityType
    description: str | None = None
    confidence: float = 1.0
    source_text: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_entity(self, document_id: str) -> Entity:
        """Convert to Entity model."""
        return Entity(
            name=self.name,
            entity_type=self.entity_type,
            description=self.description,
            properties={
                "confidence": self.confidence,
                "source_document": document_id,
                **self.properties,
            },
        )


class EntityExtractor:
    """Extracts entities from documents.

    Uses LLM-based NER combined with regex patterns for code entities.
    Supports multiple entity types and configurable extraction strategies.
    """

    # Regex patterns for code entities
    PATTERNS = {
        EntityType.FUNCTION: [
            r"(?:def|function|func)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            r"(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)",
        ],
        EntityType.CLASS: [
            r"class\s+([A-Z][a-zA-Z0-9_]*)",
            r"interface\s+([A-Z][a-zA-Z0-9_]*)",
            r"struct\s+([A-Z][a-zA-Z0-9_]*)",
        ],
        EntityType.MODULE: [
            r"(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)",
            r"require\(['\"]([^'\"]+)['\"]\)",
        ],
        EntityType.ENDPOINT: [
            r"@(?:app|router)\.(?:get|post|put|delete|patch)\(['\"]([^'\"]+)['\"]",
            r"(?:GET|POST|PUT|DELETE|PATCH)\s+(/[^\s]+)",
        ],
    }

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        """Initialize the entity extractor.

        Args:
            config: Extraction configuration
        """
        self._config = config or ExtractionConfig()
        self._llm = None

    def _get_llm(self) -> Any:
        """Lazy load LLM provider."""
        if self._llm is None:
            self._llm = get_llm_provider()
        return self._llm

    def _extract_with_regex(
        self,
        content: str,
        entity_types: list[EntityType] | None = None,
    ) -> list[ExtractedEntity]:
        """Extract entities using regex patterns.

        Args:
            content: Document content
            entity_types: Types to extract (defaults to config)

        Returns:
            List of extracted entities
        """
        types_to_extract = entity_types or self._config.entity_types
        entities: list[ExtractedEntity] = []
        seen: set[tuple[str, EntityType]] = set()

        for entity_type in types_to_extract:
            patterns = self.PATTERNS.get(entity_type, [])
            for pattern in patterns:
                for match in re.finditer(pattern, content):
                    name = match.group(1)
                    key = (name, entity_type)

                    if key not in seen:
                        seen.add(key)
                        entities.append(
                            ExtractedEntity(
                                name=name,
                                entity_type=entity_type,
                                confidence=0.8,  # Regex has lower confidence
                                source_text=match.group(0),
                            )
                        )

        return entities

    async def _extract_with_llm(
        self,
        content: str,
        entity_types: list[EntityType] | None = None,
    ) -> list[ExtractedEntity]:
        """Extract entities using LLM.

        Args:
            content: Document content
            entity_types: Types to extract (defaults to config)

        Returns:
            List of extracted entities
        """
        types_to_extract = entity_types or self._config.entity_types
        type_names = [t.value for t in types_to_extract]

        # Truncate content if too long
        max_content = 4000
        if len(content) > max_content:
            content = content[:max_content] + "\n... [truncated]"

        prompt = f"""Extract named entities from the following text. Focus on these entity types: {', '.join(type_names)}

For each entity, provide:
- name: The entity name
- type: One of {type_names}
- description: Brief description (1 sentence)
- confidence: How confident you are (0.0-1.0)

Text:
{content}

Respond with a JSON array of entities. Example format:
[
  {{"name": "FastAPI", "type": "concept", "description": "A modern Python web framework", "confidence": 0.95}}
]

If no entities found, respond with an empty array: []
Only output valid JSON, no other text."""

        try:
            llm = self._get_llm()
            response = await llm.generate(
                prompt=prompt,
                system_prompt="You are an entity extraction assistant. Extract entities and output only valid JSON.",
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )

            # Parse JSON response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                # Remove code blocks
                response = re.sub(r"```(?:json)?\n?", "", response)
                response = response.strip()

            entities_data = json.loads(response)

            entities = []
            for item in entities_data:
                try:
                    entity_type = EntityType(item.get("type", "concept"))
                    confidence = float(item.get("confidence", 0.8))

                    if confidence >= self._config.min_confidence:
                        entities.append(
                            ExtractedEntity(
                                name=item["name"],
                                entity_type=entity_type,
                                description=item.get("description"),
                                confidence=confidence,
                            )
                        )
                except (ValueError, KeyError):
                    continue

            return entities

        except (json.JSONDecodeError, Exception):
            # Fallback to empty list on error
            return []

    async def extract(
        self,
        document: Document,
        entity_types: list[EntityType] | None = None,
    ) -> list[ExtractedEntity]:
        """Extract entities from a document.

        Args:
            document: Document to extract from
            entity_types: Optional override for entity types

        Returns:
            List of extracted entities
        """
        strategy = self._config.strategy
        content = document.content

        if strategy == ExtractionStrategy.REGEX:
            return self._extract_with_regex(content, entity_types)

        elif strategy == ExtractionStrategy.LLM:
            return await self._extract_with_llm(content, entity_types)

        else:  # HYBRID
            # Combine regex and LLM extraction
            regex_entities = self._extract_with_regex(content, entity_types)
            llm_entities = await self._extract_with_llm(content, entity_types)

            # Merge, preferring LLM results for duplicates
            merged: dict[tuple[str, EntityType], ExtractedEntity] = {}

            for entity in regex_entities:
                key = (entity.name.lower(), entity.entity_type)
                merged[key] = entity

            for entity in llm_entities:
                key = (entity.name.lower(), entity.entity_type)
                # LLM results override regex (higher confidence)
                merged[key] = entity

            return list(merged.values())

    async def extract_batch(
        self,
        documents: list[Document],
        entity_types: list[EntityType] | None = None,
    ) -> dict[str, list[ExtractedEntity]]:
        """Extract entities from multiple documents.

        Args:
            documents: Documents to extract from
            entity_types: Optional override for entity types

        Returns:
            Dict mapping document IDs to extracted entities
        """
        results: dict[str, list[ExtractedEntity]] = {}

        for doc in documents:
            entities = await self.extract(doc, entity_types)
            results[doc.id] = entities

        return results

    def extract_sync(
        self,
        document: Document,
        entity_types: list[EntityType] | None = None,
    ) -> list[ExtractedEntity]:
        """Synchronous extraction (uses regex only).

        For LLM extraction, use the async extract() method.

        Args:
            document: Document to extract from
            entity_types: Optional override for entity types

        Returns:
            List of extracted entities
        """
        return self._extract_with_regex(document.content, entity_types)


def get_entity_extractor(config: ExtractionConfig | None = None) -> EntityExtractor:
    """Factory function to get an EntityExtractor instance.

    Args:
        config: Optional extraction configuration

    Returns:
        EntityExtractor instance
    """
    return EntityExtractor(config=config)
