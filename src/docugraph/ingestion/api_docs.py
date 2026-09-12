"""API documentation parsing for OpenAPI and GraphQL schemas.

Extracts structured information from API specifications:
- Endpoints, methods, parameters
- Types, schemas, models
- Descriptions and examples

Supports:
- OpenAPI 3.x (JSON/YAML)
- GraphQL SDL schemas
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from docugraph.core.models import Document


class APIDocType(StrEnum):
    """Type of API documentation."""

    OPENAPI = "openapi"
    GRAPHQL = "graphql"
    UNKNOWN = "unknown"


@dataclass
class APIParameter:
    """An API parameter (query, path, header, body)."""

    name: str
    location: str  # query, path, header, body
    param_type: str  # string, integer, boolean, etc.
    required: bool = False
    description: str | None = None
    default: Any = None
    example: Any = None


@dataclass
class APIEndpoint:
    """An API endpoint with its metadata."""

    path: str
    method: str  # GET, POST, PUT, DELETE, etc.
    summary: str | None = None
    description: str | None = None
    parameters: list[APIParameter] = field(default_factory=list)
    request_body: dict[str, Any] | None = None
    responses: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    deprecated: bool = False


@dataclass
class APISchema:
    """A data schema/model from the API spec."""

    name: str
    schema_type: str  # object, array, string, etc.
    description: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    example: Any = None


@dataclass
class GraphQLType:
    """A GraphQL type definition."""

    name: str
    kind: str  # type, input, interface, enum, union, scalar
    description: str | None = None
    fields: list[dict[str, Any]] = field(default_factory=list)
    values: list[str] | None = None  # For enums
    implements: list[str] = field(default_factory=list)


@dataclass
class ParsedAPISpec:
    """Parsed API specification."""

    doc_type: APIDocType
    title: str
    version: str | None = None
    description: str | None = None
    base_url: str | None = None
    endpoints: list[APIEndpoint] = field(default_factory=list)
    schemas: list[APISchema] = field(default_factory=list)
    graphql_types: list[GraphQLType] = field(default_factory=list)
    raw_spec: dict[str, Any] | str = field(default_factory=dict)


@dataclass
class APIDocConfig:
    """Configuration for API documentation parsing."""

    include_examples: bool = True
    include_deprecated: bool = False
    generate_markdown: bool = True
    max_description_length: int = 2000


class APIDocParser:
    """Parser for API documentation (OpenAPI, GraphQL).

    Extracts structured information and converts to indexable documents.
    """

    def __init__(self, config: APIDocConfig | None = None) -> None:
        """Initialize the API doc parser.

        Args:
            config: Parser configuration
        """
        self._config = config or APIDocConfig()

    def detect_type(self, content: str) -> APIDocType:
        """Detect the type of API documentation.

        Args:
            content: Raw content string

        Returns:
            Detected API documentation type
        """
        # Check for OpenAPI markers
        if '"openapi"' in content or "openapi:" in content:
            return APIDocType.OPENAPI
        if '"swagger"' in content or "swagger:" in content:
            return APIDocType.OPENAPI

        # Check for GraphQL SDL markers
        if re.search(r"^\s*(type|schema|query|mutation|subscription)\s+\w+", content, re.MULTILINE):
            return APIDocType.GRAPHQL

        return APIDocType.UNKNOWN

    def parse(self, content: str, source_path: str | None = None) -> ParsedAPISpec:
        """Parse API documentation content.

        Args:
            content: Raw content (JSON, YAML, or GraphQL SDL)
            source_path: Optional source file path

        Returns:
            Parsed API specification
        """
        doc_type = self.detect_type(content)

        if doc_type == APIDocType.OPENAPI:
            return self._parse_openapi(content, source_path)
        elif doc_type == APIDocType.GRAPHQL:
            return self._parse_graphql(content, source_path)
        else:
            return ParsedAPISpec(
                doc_type=APIDocType.UNKNOWN,
                title="Unknown API",
                raw_spec=content,
            )

    def parse_file(self, file_path: str | Path) -> ParsedAPISpec:
        """Parse API documentation from a file.

        Args:
            file_path: Path to the API spec file

        Returns:
            Parsed API specification
        """
        path = Path(file_path)
        content = path.read_text()
        return self.parse(content, str(path))

    def _parse_openapi(self, content: str, source_path: str | None = None) -> ParsedAPISpec:
        """Parse OpenAPI specification.

        Args:
            content: JSON or YAML content
            source_path: Source file path

        Returns:
            Parsed OpenAPI spec
        """
        # Try JSON first, then YAML
        try:
            spec = json.loads(content)
        except json.JSONDecodeError:
            try:
                import yaml

                spec = yaml.safe_load(content)
            except Exception:
                # Fallback: try to extract basic info with regex
                return self._parse_openapi_fallback(content, source_path)

        # Extract info
        info = spec.get("info", {})
        servers = spec.get("servers", [])
        base_url = servers[0].get("url") if servers else None

        # Parse endpoints
        endpoints = []
        paths = spec.get("paths", {})
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() not in (
                    "GET",
                    "POST",
                    "PUT",
                    "DELETE",
                    "PATCH",
                    "HEAD",
                    "OPTIONS",
                ):
                    continue

                if details.get("deprecated", False) and not self._config.include_deprecated:
                    continue

                # Parse parameters
                params = []
                for p in details.get("parameters", []):
                    params.append(
                        APIParameter(
                            name=p.get("name", ""),
                            location=p.get("in", "query"),
                            param_type=p.get("schema", {}).get("type", "string"),
                            required=p.get("required", False),
                            description=p.get("description"),
                            example=p.get("example") if self._config.include_examples else None,
                        )
                    )

                endpoints.append(
                    APIEndpoint(
                        path=path,
                        method=method.upper(),
                        summary=details.get("summary"),
                        description=self._truncate(details.get("description")),
                        parameters=params,
                        request_body=details.get("requestBody"),
                        responses=details.get("responses", {}),
                        tags=details.get("tags", []),
                        deprecated=details.get("deprecated", False),
                    )
                )

        # Parse schemas
        schemas = []
        components = spec.get("components", {})
        for name, schema_def in components.get("schemas", {}).items():
            schemas.append(
                APISchema(
                    name=name,
                    schema_type=schema_def.get("type", "object"),
                    description=self._truncate(schema_def.get("description")),
                    properties=schema_def.get("properties", {}),
                    required_fields=schema_def.get("required", []),
                    example=schema_def.get("example") if self._config.include_examples else None,
                )
            )

        return ParsedAPISpec(
            doc_type=APIDocType.OPENAPI,
            title=info.get("title", "API"),
            version=info.get("version"),
            description=self._truncate(info.get("description")),
            base_url=base_url,
            endpoints=endpoints,
            schemas=schemas,
            raw_spec=spec,
        )

    def _parse_openapi_fallback(self, content: str, _source_path: str | None) -> ParsedAPISpec:
        """Fallback OpenAPI parsing using regex.

        Args:
            content: Raw content
            source_path: Source file path

        Returns:
            Basic parsed spec
        """
        # Extract title
        title_match = re.search(r'"title"\s*:\s*"([^"]+)"', content)
        title = title_match.group(1) if title_match else "API"

        # Extract version
        version_match = re.search(r'"version"\s*:\s*"([^"]+)"', content)
        version = version_match.group(1) if version_match else None

        # Extract paths (basic)
        endpoints = []
        path_pattern = r'"(/[^"]+)"\s*:\s*\{'
        for match in re.finditer(path_pattern, content):
            path = match.group(1)
            # Look for methods after this path
            for method in ["get", "post", "put", "delete", "patch"]:
                if f'"{method}"' in content[match.start() : match.start() + 500]:
                    endpoints.append(
                        APIEndpoint(
                            path=path,
                            method=method.upper(),
                        )
                    )

        return ParsedAPISpec(
            doc_type=APIDocType.OPENAPI,
            title=title,
            version=version,
            endpoints=endpoints,
            raw_spec=content,
        )

    def _parse_graphql(self, content: str, _source_path: str | None = None) -> ParsedAPISpec:
        """Parse GraphQL SDL schema.

        Args:
            content: GraphQL SDL content
            source_path: Source file path

        Returns:
            Parsed GraphQL spec
        """
        types: list[GraphQLType] = []

        # Parse type definitions
        type_pattern = r'(?:"""([^"]*)"""\s*)?(type|input|interface|enum|union|scalar)\s+(\w+)(?:\s+implements\s+([\w\s&]+))?(?:\s*\{([^}]*)\})?'

        for match in re.finditer(type_pattern, content, re.DOTALL):
            description = match.group(1)
            kind = match.group(2)
            name = match.group(3)
            implements = match.group(4)
            body = match.group(5)

            gql_type = GraphQLType(
                name=name,
                kind=kind,
                description=self._truncate(description.strip()) if description else None,
                implements=implements.split("&") if implements else [],
            )

            if kind == "enum" and body:
                # Parse enum values
                values = re.findall(r"(\w+)", body)
                gql_type.values = values
            elif body:
                # Parse fields
                field_pattern = r'(?:"""([^"]*)"""\s*)?(\w+)(?:\([^)]*\))?\s*:\s*([^\n!]+!?)'
                for field_match in re.finditer(field_pattern, body):
                    gql_type.fields.append(
                        {
                            "description": field_match.group(1).strip()
                            if field_match.group(1)
                            else None,
                            "name": field_match.group(2),
                            "type": field_match.group(3).strip(),
                        }
                    )

            types.append(gql_type)

        # Extract schema title from comments or first type
        title = "GraphQL API"
        title_match = re.search(r"#\s*(.+)", content)
        if title_match:
            title = title_match.group(1).strip()

        return ParsedAPISpec(
            doc_type=APIDocType.GRAPHQL,
            title=title,
            graphql_types=types,
            raw_spec=content,
        )

    def _truncate(self, text: str | None) -> str | None:
        """Truncate text to max length.

        Args:
            text: Text to truncate

        Returns:
            Truncated text
        """
        if not text:
            return text
        if len(text) <= self._config.max_description_length:
            return text
        return text[: self._config.max_description_length - 3] + "..."

    def to_documents(self, spec: ParsedAPISpec, source_path: str | None = None) -> list[Document]:
        """Convert parsed spec to indexable documents.

        Args:
            spec: Parsed API specification
            source_path: Source file path

        Returns:
            List of Document objects for indexing
        """
        documents: list[Document] = []
        base_metadata = {
            "doc_type": spec.doc_type.value,
            "api_title": spec.title,
            "api_version": spec.version,
            "source_path": source_path,
        }

        if spec.doc_type == APIDocType.OPENAPI:
            # Create document for each endpoint
            for endpoint in spec.endpoints:
                content = self._endpoint_to_markdown(endpoint, spec)
                documents.append(
                    Document(
                        content=content,
                        content_type="api_endpoint",
                        source_path=source_path,
                        metadata={
                            **base_metadata,
                            "endpoint_path": endpoint.path,
                            "endpoint_method": endpoint.method,
                            "tags": endpoint.tags,
                        },
                    )
                )

            # Create document for each schema
            for schema in spec.schemas:
                content = self._schema_to_markdown(schema)
                documents.append(
                    Document(
                        content=content,
                        content_type="api_schema",
                        source_path=source_path,
                        metadata={
                            **base_metadata,
                            "schema_name": schema.name,
                            "schema_type": schema.schema_type,
                        },
                    )
                )

        elif spec.doc_type == APIDocType.GRAPHQL:
            # Create document for each GraphQL type
            for gql_type in spec.graphql_types:
                content = self._graphql_type_to_markdown(gql_type)
                documents.append(
                    Document(
                        content=content,
                        content_type="graphql_type",
                        source_path=source_path,
                        metadata={
                            **base_metadata,
                            "type_name": gql_type.name,
                            "type_kind": gql_type.kind,
                        },
                    )
                )

        # Add overview document
        if spec.description or spec.endpoints or spec.graphql_types:
            overview = self._spec_to_overview(spec)
            documents.insert(
                0,
                Document(
                    content=overview,
                    content_type="api_overview",
                    source_path=source_path,
                    metadata={
                        **base_metadata,
                        "is_overview": True,
                    },
                ),
            )

        return documents

    def _endpoint_to_markdown(self, endpoint: APIEndpoint, _spec: ParsedAPISpec) -> str:
        """Convert endpoint to markdown.

        Args:
            endpoint: API endpoint
            spec: Full spec for context

        Returns:
            Markdown representation
        """
        parts = [f"## {endpoint.method} {endpoint.path}"]

        if endpoint.summary:
            parts.append(f"\n{endpoint.summary}")

        if endpoint.description:
            parts.append(f"\n{endpoint.description}")

        if endpoint.tags:
            parts.append(f"\n**Tags:** {', '.join(endpoint.tags)}")

        if endpoint.deprecated:
            parts.append("\n**DEPRECATED**")

        if endpoint.parameters:
            parts.append("\n### Parameters")
            for param in endpoint.parameters:
                required = " (required)" if param.required else ""
                desc = f" - {param.description}" if param.description else ""
                parts.append(
                    f"- `{param.name}` ({param.location}, {param.param_type}){required}{desc}"
                )

        if endpoint.request_body:
            parts.append("\n### Request Body")
            content = endpoint.request_body.get("content", {})
            for media_type, details in content.items():
                parts.append(f"**Content-Type:** {media_type}")
                if "schema" in details:
                    schema_ref = details["schema"].get("$ref", "")
                    if schema_ref:
                        schema_name = schema_ref.split("/")[-1]
                        parts.append(f"**Schema:** {schema_name}")

        if endpoint.responses:
            parts.append("\n### Responses")
            for status, details in endpoint.responses.items():
                desc = details.get("description", "") if isinstance(details, dict) else ""
                parts.append(f"- **{status}**: {desc}")

        return "\n".join(parts)

    def _schema_to_markdown(self, schema: APISchema) -> str:
        """Convert schema to markdown.

        Args:
            schema: API schema

        Returns:
            Markdown representation
        """
        parts = [f"## Schema: {schema.name}"]

        if schema.description:
            parts.append(f"\n{schema.description}")

        parts.append(f"\n**Type:** {schema.schema_type}")

        if schema.properties:
            parts.append("\n### Properties")
            for prop_name, prop_details in schema.properties.items():
                required = " (required)" if prop_name in schema.required_fields else ""
                prop_type = (
                    prop_details.get("type", "any") if isinstance(prop_details, dict) else "any"
                )
                desc = prop_details.get("description", "") if isinstance(prop_details, dict) else ""
                parts.append(f"- `{prop_name}` ({prop_type}){required}: {desc}")

        if schema.example and self._config.include_examples:
            parts.append("\n### Example")
            parts.append(f"```json\n{json.dumps(schema.example, indent=2)}\n```")

        return "\n".join(parts)

    def _graphql_type_to_markdown(self, gql_type: GraphQLType) -> str:
        """Convert GraphQL type to markdown.

        Args:
            gql_type: GraphQL type

        Returns:
            Markdown representation
        """
        parts = [f"## {gql_type.kind.title()}: {gql_type.name}"]

        if gql_type.description:
            parts.append(f"\n{gql_type.description}")

        if gql_type.implements:
            parts.append(f"\n**Implements:** {', '.join(gql_type.implements)}")

        if gql_type.kind == "enum" and gql_type.values:
            parts.append("\n### Values")
            for value in gql_type.values:
                parts.append(f"- `{value}`")
        elif gql_type.fields:
            parts.append("\n### Fields")
            for field in gql_type.fields:
                desc = f" - {field['description']}" if field.get("description") else ""
                parts.append(f"- `{field['name']}`: {field['type']}{desc}")

        return "\n".join(parts)

    def _spec_to_overview(self, spec: ParsedAPISpec) -> str:
        """Create overview document from spec.

        Args:
            spec: Parsed API spec

        Returns:
            Overview markdown
        """
        parts = [f"# {spec.title}"]

        if spec.version:
            parts.append(f"\n**Version:** {spec.version}")

        if spec.description:
            parts.append(f"\n{spec.description}")

        if spec.base_url:
            parts.append(f"\n**Base URL:** {spec.base_url}")

        if spec.endpoints:
            parts.append("\n## Endpoints")
            for endpoint in spec.endpoints[:20]:  # Limit for overview
                parts.append(f"- `{endpoint.method} {endpoint.path}`: {endpoint.summary or ''}")

        if spec.schemas:
            parts.append("\n## Schemas")
            for schema in spec.schemas[:20]:
                parts.append(f"- `{schema.name}` ({schema.schema_type})")

        if spec.graphql_types:
            parts.append("\n## Types")
            for gql_type in spec.graphql_types[:20]:
                parts.append(f"- `{gql_type.name}` ({gql_type.kind})")

        return "\n".join(parts)


def get_api_doc_parser(config: APIDocConfig | None = None) -> APIDocParser:
    """Factory function to get an APIDocParser instance.

    Args:
        config: Optional parser configuration

    Returns:
        APIDocParser instance
    """
    return APIDocParser(config=config)
