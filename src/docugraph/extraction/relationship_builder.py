"""Relationship inference between entities.

Builds relationships from:
- Code analysis (imports, calls, inheritance)
- Document analysis (references, links)
- LLM-based inference for semantic relationships

LLM-agnostic: Works with Ollama (local), OpenAI, or Anthropic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from docugraph.core.llm import get_llm_provider
from docugraph.core.models import Entity, Relationship, RelationshipType
from docugraph.extraction.code_parser import CodeParser, ParsedCode
from docugraph.extraction.entity_extractor import ExtractedEntity


@dataclass
class InferredRelationship:
    """An inferred relationship between entities."""

    source_name: str
    target_name: str
    relationship_type: RelationshipType
    confidence: float = 1.0
    evidence: str | None = None  # Text that supports this relationship
    properties: dict[str, Any] = field(default_factory=dict)

    def to_relationship(
        self,
        source_id: str,
        target_id: str,
    ) -> Relationship:
        """Convert to Relationship model."""
        return Relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=self.relationship_type,
            properties={
                "confidence": self.confidence,
                "evidence": self.evidence,
                **self.properties,
            },
        )


@dataclass
class RelationshipBuilderConfig:
    """Configuration for relationship building."""

    # Enable/disable inference methods
    use_code_analysis: bool = True
    use_document_analysis: bool = True
    use_llm_inference: bool = False  # Expensive, disabled by default

    # LLM settings
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000

    # Confidence thresholds
    min_confidence: float = 0.5


class RelationshipBuilder:
    """Builds and infers relationships between entities.

    Combines multiple strategies:
    1. Code analysis: Import, inheritance, composition relationships
    2. Document analysis: Reference and link-based relationships
    3. LLM inference: Semantic relationships from context
    """

    def __init__(self, config: RelationshipBuilderConfig | None = None) -> None:
        """Initialize the relationship builder.

        Args:
            config: Builder configuration
        """
        self._config = config or RelationshipBuilderConfig()
        self._code_parser = CodeParser()
        self._llm = None

    def _get_llm(self) -> Any:
        """Lazy load LLM provider."""
        if self._llm is None:
            self._llm = get_llm_provider()
        return self._llm

    def from_code(self, parsed_code: ParsedCode) -> list[InferredRelationship]:
        """Infer relationships from parsed code.

        Extracts:
        - IMPORTS: Module import relationships
        - INHERITS: Class inheritance
        - CONTAINS: Class/module contains methods/functions
        - CALLS: Function/method calls (basic detection)

        Args:
            parsed_code: Parsed code structure

        Returns:
            List of inferred relationships
        """
        relationships: list[InferredRelationship] = []

        # Get module name from file path
        module_name = parsed_code.file_path.replace("/", ".").replace("\\", ".")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]

        # Import relationships
        for imp in parsed_code.imports:
            relationships.append(
                InferredRelationship(
                    source_name=module_name,
                    target_name=imp.module,
                    relationship_type=RelationshipType.IMPORTS,
                    confidence=1.0,
                    evidence=f"import at line {imp.line}",
                )
            )

        # Class contains methods
        for symbol in parsed_code.symbols:
            if symbol.symbol_type == "class":
                # Find methods of this class
                for other in parsed_code.symbols:
                    if other.parent == symbol.name and other.symbol_type == "method":
                        relationships.append(
                            InferredRelationship(
                                source_name=symbol.name,
                                target_name=other.name,
                                relationship_type=RelationshipType.DESCRIBES,
                                confidence=1.0,
                                evidence=f"method defined at line {other.start_line}",
                            )
                        )

        # Detect inheritance from class content (basic)
        content = parsed_code.raw_content or ""
        for symbol in parsed_code.symbols:
            if symbol.symbol_type == "class":
                # Look for class ClassName(BaseClass):
                pattern = rf"class\s+{re.escape(symbol.name)}\s*\(([^)]+)\)"
                match = re.search(pattern, content)
                if match:
                    bases = match.group(1).split(",")
                    for base in bases:
                        base = base.strip()
                        if base and base not in ("object", "ABC"):
                            relationships.append(
                                InferredRelationship(
                                    source_name=symbol.name,
                                    target_name=base,
                                    relationship_type=RelationshipType.INHERITS,
                                    confidence=1.0,
                                    evidence=f"class definition at line {symbol.start_line}",
                                )
                            )

        return relationships

    def from_document_content(
        self,
        content: str,
        entities: list[ExtractedEntity],
    ) -> list[InferredRelationship]:
        """Infer relationships from document content and extracted entities.

        Looks for:
        - Proximity-based relationships
        - Explicit mentions ("X uses Y", "X depends on Y")
        - Links and references

        Args:
            content: Document content
            entities: Entities extracted from the document

        Returns:
            List of inferred relationships
        """
        relationships: list[InferredRelationship] = []
        entity_names = {e.name for e in entities}

        # Relationship indicator patterns
        # Note: For "X uses Y", we create "Y USED_BY X" (inverted)
        patterns = [
            (r"(\w+)\s+(?:depends\s+on|relies\s+on)\s+(\w+)", RelationshipType.DEPENDS_ON),
            (r"(\w+)\s+(?:extends|inherits\s+from)\s+(\w+)", RelationshipType.INHERITS),
            (r"(\w+)\s+(?:implements)\s+(\w+)", RelationshipType.IMPLEMENTS),
            (r"(\w+)\s+(?:describes|documents)\s+(\w+)", RelationshipType.DESCRIBES),
            (r"(\w+)\s+(?:is\s+related\s+to|relates\s+to)\s+(\w+)", RelationshipType.RELATED_TO),
            (r"(\w+)\s+(?:calls|invokes)\s+(\w+)", RelationshipType.CALLS),
            (r"(\w+)\s+(?:supersedes|replaces)\s+(\w+)", RelationshipType.SUPERSEDES),
            (r"(\w+)\s+(?:deprecates)\s+(\w+)", RelationshipType.DEPRECATES),
        ]

        for pattern, rel_type in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                source = match.group(1)
                target = match.group(2)

                # Only create relationship if both entities are known
                if source in entity_names and target in entity_names:
                    relationships.append(
                        InferredRelationship(
                            source_name=source,
                            target_name=target,
                            relationship_type=rel_type,
                            confidence=0.7,
                            evidence=match.group(0),
                        )
                    )

        # Co-occurrence relationships (entities mentioned near each other)
        entity_list = list(entity_names)
        for i, entity1 in enumerate(entity_list):
            for entity2 in entity_list[i + 1 :]:
                # Check if they appear in same paragraph
                for para in content.split("\n\n"):
                    if entity1 in para and entity2 in para:
                        relationships.append(
                            InferredRelationship(
                                source_name=entity1,
                                target_name=entity2,
                                relationship_type=RelationshipType.RELATED_TO,
                                confidence=0.5,
                                evidence=f"Co-occurrence in paragraph",
                            )
                        )
                        break  # Only count once per pair

        return relationships

    async def from_llm_inference(
        self,
        content: str,
        entities: list[ExtractedEntity],
    ) -> list[InferredRelationship]:
        """Infer relationships using LLM.

        Uses the LLM to identify semantic relationships that may not
        be explicitly stated in the text.

        Args:
            content: Document content
            entities: Entities to find relationships between

        Returns:
            List of inferred relationships
        """
        if not entities or len(entities) < 2:
            return []

        # Truncate content if too long
        max_content = 3000
        if len(content) > max_content:
            content = content[:max_content] + "\n... [truncated]"

        # Build entity list for prompt
        entity_list = "\n".join(
            f"- {e.name} ({e.entity_type.value}): {e.description or 'No description'}"
            for e in entities[:20]  # Limit entities
        )

        # Relationship types for prompt
        rel_types = [
            "CALLS - Function/method calls another",
            "IMPORTS - Module imports another",
            "INHERITS - Class inherits from another",
            "IMPLEMENTS - Class implements interface",
            "DEPENDS_ON - Entity depends on another",
            "USED_BY - Entity is used by another",
            "DESCRIBES - Entity describes another",
            "RELATED_TO - Entities are related",
        ]

        prompt = f"""Analyze the relationships between entities in the following text.

Entities found:
{entity_list}

Relationship types to consider:
{chr(10).join(rel_types)}

Text:
{content}

Identify relationships between the entities. For each relationship:
- source: Source entity name
- target: Target entity name
- type: Relationship type (one of: CALLS, IMPORTS, INHERITS, IMPLEMENTS, DEPENDS_ON, USED_BY, DESCRIBES, RELATED_TO)
- confidence: Your confidence (0.0-1.0)
- evidence: Brief evidence from the text

Respond with a JSON array. Example:
[
  {{"source": "FastAPI", "target": "Starlette", "type": "USES", "confidence": 0.9, "evidence": "FastAPI is built on top of Starlette"}}
]

If no relationships found, respond with: []
Only output valid JSON."""

        try:
            llm = self._get_llm()
            response = await llm.generate(
                prompt=prompt,
                system_prompt="You are a relationship extraction assistant. Identify relationships between entities and output only valid JSON.",
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
            )

            # Parse JSON response
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r"```(?:json)?\n?", "", response)
                response = response.strip()

            relations_data = json.loads(response)

            relationships: list[InferredRelationship] = []
            type_map = {
                "CALLS": RelationshipType.CALLS,
                "IMPORTS": RelationshipType.IMPORTS,
                "INHERITS": RelationshipType.INHERITS,
                "IMPLEMENTS": RelationshipType.IMPLEMENTS,
                "DEPENDS_ON": RelationshipType.DEPENDS_ON,
                "USED_BY": RelationshipType.USED_BY,
                "DESCRIBES": RelationshipType.DESCRIBES,
                "RELATED_TO": RelationshipType.RELATED_TO,
                "EXAMPLE_OF": RelationshipType.EXAMPLE_OF,
                "SUPERSEDES": RelationshipType.SUPERSEDES,
                "DEPRECATES": RelationshipType.DEPRECATES,
            }

            for item in relations_data:
                try:
                    rel_type = type_map.get(item.get("type", "").upper(), RelationshipType.RELATED_TO)
                    confidence = float(item.get("confidence", 0.7))

                    if confidence >= self._config.min_confidence:
                        relationships.append(
                            InferredRelationship(
                                source_name=item["source"],
                                target_name=item["target"],
                                relationship_type=rel_type,
                                confidence=confidence,
                                evidence=item.get("evidence"),
                            )
                        )
                except (ValueError, KeyError):
                    continue

            return relationships

        except (json.JSONDecodeError, Exception):
            return []

    async def build_relationships(
        self,
        content: str,
        entities: list[ExtractedEntity],
        parsed_code: ParsedCode | None = None,
    ) -> list[InferredRelationship]:
        """Build relationships using all configured methods.

        Args:
            content: Document/code content
            entities: Extracted entities
            parsed_code: Optional parsed code structure

        Returns:
            Combined list of inferred relationships
        """
        relationships: list[InferredRelationship] = []

        # Code analysis
        if self._config.use_code_analysis and parsed_code:
            code_rels = self.from_code(parsed_code)
            relationships.extend(code_rels)

        # Document analysis
        if self._config.use_document_analysis:
            doc_rels = self.from_document_content(content, entities)
            relationships.extend(doc_rels)

        # LLM inference
        if self._config.use_llm_inference:
            llm_rels = await self.from_llm_inference(content, entities)
            relationships.extend(llm_rels)

        # Deduplicate relationships
        seen: set[tuple[str, str, RelationshipType]] = set()
        deduped: list[InferredRelationship] = []

        for rel in relationships:
            key = (rel.source_name, rel.target_name, rel.relationship_type)
            if key not in seen:
                seen.add(key)
                deduped.append(rel)
            else:
                # Update confidence if we see the same relationship again
                for existing in deduped:
                    if (existing.source_name, existing.target_name, existing.relationship_type) == key:
                        existing.confidence = max(existing.confidence, rel.confidence)
                        break

        return deduped

    def build_relationships_sync(
        self,
        content: str,
        entities: list[ExtractedEntity],
        parsed_code: ParsedCode | None = None,
    ) -> list[InferredRelationship]:
        """Synchronous relationship building (no LLM).

        Args:
            content: Document/code content
            entities: Extracted entities
            parsed_code: Optional parsed code structure

        Returns:
            List of inferred relationships
        """
        relationships: list[InferredRelationship] = []

        if self._config.use_code_analysis and parsed_code:
            relationships.extend(self.from_code(parsed_code))

        if self._config.use_document_analysis:
            relationships.extend(self.from_document_content(content, entities))

        # Deduplicate
        seen: set[tuple[str, str, RelationshipType]] = set()
        deduped: list[InferredRelationship] = []

        for rel in relationships:
            key = (rel.source_name, rel.target_name, rel.relationship_type)
            if key not in seen:
                seen.add(key)
                deduped.append(rel)

        return deduped


def get_relationship_builder(
    config: RelationshipBuilderConfig | None = None,
) -> RelationshipBuilder:
    """Factory function to get a RelationshipBuilder instance.

    Args:
        config: Optional builder configuration

    Returns:
        RelationshipBuilder instance
    """
    return RelationshipBuilder(config=config)
