"""Unit tests for the extraction module."""

import pytest

from docugraph.core.models import Document, EntityType, RelationshipType
from docugraph.extraction.entity_extractor import (
    EntityExtractor,
    ExtractionConfig,
    ExtractionStrategy,
    ExtractedEntity,
    get_entity_extractor,
)
from docugraph.extraction.code_parser import (
    CodeParser,
    Language,
    ParsedCode,
    get_code_parser,
)
from docugraph.extraction.relationship_builder import (
    RelationshipBuilder,
    RelationshipBuilderConfig,
    InferredRelationship,
    get_relationship_builder,
)


class TestExtractionConfig:
    """Tests for ExtractionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ExtractionConfig()
        assert config.strategy == ExtractionStrategy.HYBRID
        assert config.min_confidence == 0.5

    def test_custom_strategy(self):
        """Test custom strategy configuration."""
        config = ExtractionConfig(strategy=ExtractionStrategy.REGEX)
        assert config.strategy == ExtractionStrategy.REGEX


class TestExtractedEntity:
    """Tests for ExtractedEntity."""

    def test_extracted_entity_creation(self):
        """Test basic entity creation."""
        entity = ExtractedEntity(
            name="MyClass",
            entity_type=EntityType.CLASS,
            confidence=0.9,
        )
        assert entity.name == "MyClass"
        assert entity.entity_type == EntityType.CLASS
        assert entity.confidence == 0.9

    def test_to_entity_conversion(self):
        """Test conversion to Entity model."""
        extracted = ExtractedEntity(
            name="process",
            entity_type=EntityType.FUNCTION,
            description="Process data",
            confidence=0.85,
        )
        entity = extracted.to_entity("doc-123")
        assert entity.name == "process"
        assert entity.entity_type == EntityType.FUNCTION
        assert entity.properties["confidence"] == 0.85
        assert entity.properties["source_document"] == "doc-123"


class TestEntityExtractor:
    """Tests for EntityExtractor."""

    def test_extractor_creation(self):
        """Test extractor instantiation."""
        extractor = EntityExtractor()
        assert extractor is not None

    def test_regex_extraction_functions(self):
        """Test regex extraction of functions."""
        config = ExtractionConfig(strategy=ExtractionStrategy.REGEX)
        extractor = EntityExtractor(config=config)

        content = """
def my_function():
    pass

def another_function(arg1, arg2):
    return arg1 + arg2
"""
        doc = Document(content=content, content_type="code")
        entities = extractor.extract_sync(doc, [EntityType.FUNCTION])

        names = [e.name for e in entities]
        assert "my_function" in names
        assert "another_function" in names

    def test_regex_extraction_classes(self):
        """Test regex extraction of classes."""
        config = ExtractionConfig(strategy=ExtractionStrategy.REGEX)
        extractor = EntityExtractor(config=config)

        content = """
class MyClass:
    pass

class AnotherClass(BaseClass):
    def method(self):
        pass
"""
        doc = Document(content=content, content_type="code")
        entities = extractor.extract_sync(doc, [EntityType.CLASS])

        names = [e.name for e in entities]
        assert "MyClass" in names
        assert "AnotherClass" in names

    def test_regex_extraction_modules(self):
        """Test regex extraction of module imports."""
        config = ExtractionConfig(strategy=ExtractionStrategy.REGEX)
        extractor = EntityExtractor(config=config)

        content = """
import os
from pathlib import Path
import json
from collections import defaultdict
"""
        doc = Document(content=content, content_type="code")
        entities = extractor.extract_sync(doc, [EntityType.MODULE])

        names = [e.name for e in entities]
        assert "os" in names
        assert "pathlib" in names
        assert "json" in names
        assert "collections" in names

    def test_extraction_confidence(self):
        """Test that regex entities have appropriate confidence."""
        config = ExtractionConfig(strategy=ExtractionStrategy.REGEX)
        extractor = EntityExtractor(config=config)

        content = "def test_func(): pass"
        doc = Document(content=content, content_type="code")
        entities = extractor.extract_sync(doc, [EntityType.FUNCTION])

        assert len(entities) == 1
        assert entities[0].confidence == 0.8  # Regex confidence


class TestCodeParser:
    """Tests for CodeParser."""

    def test_parser_creation(self):
        """Test parser instantiation."""
        parser = CodeParser()
        assert parser is not None

    def test_language_detection_python(self):
        """Test Python file detection."""
        parser = CodeParser()
        lang = parser.detect_language("test.py")
        assert lang == Language.PYTHON

    def test_language_detection_javascript(self):
        """Test JavaScript file detection."""
        parser = CodeParser()
        assert parser.detect_language("test.js") == Language.JAVASCRIPT
        assert parser.detect_language("test.ts") == Language.TYPESCRIPT

    def test_parse_python(self):
        """Test parsing Python code."""
        parser = CodeParser()
        code = """
import os
from pathlib import Path

class MyClass:
    def __init__(self):
        pass

    def method(self):
        return True

def standalone_function():
    pass
"""
        result = parser.parse(code, "test.py")

        assert result.language == Language.PYTHON
        assert len(result.imports) >= 2
        assert len(result.symbols) >= 3  # class + 2 methods or function

        # Check imports
        import_modules = [i.module for i in result.imports]
        assert "os" in import_modules
        assert "pathlib" in import_modules

        # Check symbols
        symbol_names = [s.name for s in result.symbols]
        assert "MyClass" in symbol_names

    def test_parse_javascript(self):
        """Test parsing JavaScript code."""
        parser = CodeParser()
        code = """
import React from 'react';
const axios = require('axios');

class Component extends React.Component {
    render() {
        return null;
    }
}

function helper() {
    return true;
}

const arrowFunc = () => {};
"""
        result = parser.parse(code, "test.js")

        assert result.language == Language.JAVASCRIPT
        assert len(result.symbols) >= 1

    def test_parse_unknown_language(self):
        """Test parsing unknown file type."""
        parser = CodeParser()
        result = parser.parse("some content", "test.xyz")
        assert result.language == Language.UNKNOWN


class TestRelationshipBuilder:
    """Tests for RelationshipBuilder."""

    def test_builder_creation(self):
        """Test builder instantiation."""
        builder = RelationshipBuilder()
        assert builder is not None

    def test_from_code_imports(self):
        """Test relationship inference from imports."""
        parser = CodeParser()
        builder = RelationshipBuilder()

        code = """
import os
from pathlib import Path
"""
        parsed = parser.parse(code, "mymodule.py")
        relationships = builder.from_code(parsed)

        # Should have IMPORTS relationships
        import_rels = [r for r in relationships if r.relationship_type == RelationshipType.IMPORTS]
        assert len(import_rels) >= 2

    def test_from_document_content(self):
        """Test relationship inference from document content."""
        builder = RelationshipBuilder()

        content = "Django depends on Python. FastAPI inherits from Starlette."
        entities = [
            ExtractedEntity(name="FastAPI", entity_type=EntityType.CONCEPT),
            ExtractedEntity(name="Starlette", entity_type=EntityType.CONCEPT),
            ExtractedEntity(name="Django", entity_type=EntityType.CONCEPT),
            ExtractedEntity(name="Python", entity_type=EntityType.CONCEPT),
        ]

        relationships = builder.from_document_content(content, entities)

        # Should find DEPENDS_ON and/or RELATED_TO relationships
        assert len(relationships) > 0

    def test_inferred_relationship_conversion(self):
        """Test InferredRelationship to Relationship conversion."""
        inferred = InferredRelationship(
            source_name="ClassA",
            target_name="ClassB",
            relationship_type=RelationshipType.INHERITS,
            confidence=0.9,
            evidence="class ClassA(ClassB)",
        )

        relationship = inferred.to_relationship("id-a", "id-b")
        assert relationship.source_id == "id-a"
        assert relationship.target_id == "id-b"
        assert relationship.relationship_type == RelationshipType.INHERITS
        assert relationship.properties["confidence"] == 0.9


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_entity_extractor(self):
        """Test entity extractor factory."""
        extractor = get_entity_extractor()
        assert isinstance(extractor, EntityExtractor)

    def test_get_code_parser(self):
        """Test code parser factory."""
        parser = get_code_parser()
        assert isinstance(parser, CodeParser)

    def test_get_relationship_builder(self):
        """Test relationship builder factory."""
        builder = get_relationship_builder()
        assert isinstance(builder, RelationshipBuilder)
