"""Unit tests for the resilient Graphiti LLM client (Ollama path).

Covers the JSON-shape tolerance that makes local/Ollama-served models work
with Graphiti's strict ``Model(**json.loads(text))`` extraction path:
  - bare-list responses coerced into the schema's list field
  - loose item keys (e.g. ``entity_name``) normalized to schema field names
  - prose-wrapped JSON extracted before parsing

These exercise the client's pure helpers with a fake response_model — no
network and no real LLM.
"""

from pydantic import BaseModel

from docugraph.core.llm import OllamaLLM


class _Entity(BaseModel):
    name: str
    entity_type_id: int = 0


class _ExtractedEntities(BaseModel):
    extracted_entities: list[_Entity]


def _client():
    # Construct the resilient Graphiti client without touching the network.
    return OllamaLLM(model="fake", base_url="http://localhost:11434").get_graphiti_client()


class TestCoerceToSchema:
    def test_bare_list_wrapped_into_list_field(self):
        client = _client()
        out = client._coerce_to_schema([{"name": "FastAPI"}], _ExtractedEntities)
        assert out == {"extracted_entities": [{"name": "FastAPI"}]}

    def test_loose_item_keys_normalized(self):
        client = _client()
        # Model emitted `entity_name` where the schema wants `name`.
        raw = {"extracted_entities": [{"entity_name": "FastAPI", "entity_type_id": 0}]}
        out = client._coerce_to_schema(raw, _ExtractedEntities)
        assert out["extracted_entities"][0]["name"] == "FastAPI"
        assert "entity_name" not in out["extracted_entities"][0]
        # Validates against the real schema now.
        _ExtractedEntities(**out)

    def test_mapping_passthrough(self):
        client = _client()
        raw = {"extracted_entities": [{"name": "X", "entity_type_id": 1}]}
        assert client._coerce_to_schema(raw, _ExtractedEntities) == raw

    def test_exact_field_not_disturbed(self):
        client = _client()
        # A field that already matches must be left alone even if another key
        # could suffix-match it.
        raw = {"extracted_entities": [{"name": "X"}]}
        out = client._coerce_to_schema(raw, _ExtractedEntities)
        assert out["extracted_entities"][0]["name"] == "X"


class TestLoadsTolerant:
    def test_plain_json(self):
        client = _client()
        assert client._loads_tolerant('{"a": 1}') == {"a": 1}

    def test_json_embedded_in_prose(self):
        client = _client()
        text = 'Here is the result:\n{"extracted_entities": []}\nHope that helps!'
        assert client._loads_tolerant(text) == {"extracted_entities": []}

    def test_bare_list_in_prose(self):
        client = _client()
        text = 'Sure! [\n  {"name": "FastAPI"}\n]'
        assert client._loads_tolerant(text) == [{"name": "FastAPI"}]


class TestSchemaInjection:
    def test_schema_travels_in_the_prompt(self):
        """The schema must reach a model that ignores ``response_format``.

        Measured on deepseek-v4.1-flash:cloud through Ollama: both the native
        ``/api/chat`` ``format`` field and the OpenAI-compatible
        ``response_format`` are ignored, and the extraction prompt came back as
        a bare list of ``{name, type}`` objects -- no ``entity_type_id``, so
        Graphiti's ``Model(**json.loads(...))`` raised. Graphiti's own fix is
        ``structured_output_mode='json_object'``, which appends the response
        model's schema to the last message. This client must select that mode.
        """
        import asyncio

        from graphiti_core.prompts.models import Message

        client = _client()
        assert client.structured_output_mode == "json_object"
        captured = {}

        async def fake_generate(messages, response_model=None, max_tokens=None, model_size=None):
            captured["last"] = messages[-1].content
            return {"extracted_entities": []}

        client._generate_response = fake_generate  # type: ignore[method-assign]
        out = asyncio.run(
            client.generate_response(
                [Message(role="system", content="sys"), Message(role="user", content="user")],
                response_model=_ExtractedEntities,
            )
        )

        assert out == {"extracted_entities": []}
        assert "extracted_entities" in captured["last"]
        assert "entity_type_id" in captured["last"]
