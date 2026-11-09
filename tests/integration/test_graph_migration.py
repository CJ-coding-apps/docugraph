"""Migration: one engine per process, counts verified, nothing overwritten.

ORACLE: WP-docugraph-0.4.0 §3.3, §5 M1–M3, §10.2 D-8, §10.4 P0-5,
§6 (load both engines in one process must show up as a guard failure).

The tests inspect the migrated graph from a child process that imports only the
engine it asks about, for the same reason the command does: the two engines are
never loaded together.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from docugraph.storage.graph_migration import MigrationError, strip_fts_definitions


class FakeEmbedder:
    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]

    def embed_query(self, query: str) -> list[float]:  # noqa: ARG002
        return [0.1] * self._dim

    @property
    def dimensions(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fake-graph-embedder"


def _fake_llm_client() -> Any:
    from graphiti_core.llm_client.client import LLMClient
    from graphiti_core.llm_client.config import LLMConfig

    class FakeGraphitiLLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(config=LLMConfig(model="fake", small_model="fake"))

        async def _generate_response(
            self,
            messages: Any,
            response_model: Any = None,
            max_tokens: Any = None,
            model_size: Any = None,
        ) -> Any:
            if response_model is None:
                return {}
            return {
                name: []
                for name, field in response_model.model_fields.items()
                if field.annotation is not None and "list" in str(field.annotation).lower()
            }

    return FakeGraphitiLLM()


def _run_engine(engine: str, body: str, env_extra: dict[str, str] | None = None) -> dict:
    """Run `body` in a child that imports only `engine` and returns JSON."""
    other = "ladybug" if engine == "kuzu" else "kuzu"
    script = f"""
import json
import sys

import {engine}

assert "{other}" not in sys.modules, "the {engine} child loaded {other}"

{body}
"""
    env = os.environ.copy()
    env.update(env_extra or {})
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _counts(engine: str, db_path: Path, fts: Path | None = None) -> dict[str, int]:
    node_tables = ("Episodic", "Entity", "Community", "RelatesToNode_", "Saga")
    rel_tables = ("RELATES_TO", "MENTIONS", "HAS_MEMBER", "HAS_EPISODE", "NEXT_EPISODE")
    lines = [
        f"db = {engine}.Database({json.dumps(str(db_path))})",
        f"conn = {engine}.Connection(db)",
    ]
    if engine == "ladybug" and fts is not None:
        lines.append(f"conn.execute(\"LOAD EXTENSION '{fts}';\")")
    lines.append("counts = {}")
    for table in node_tables:
        lines.append(
            f"counts[{table!r}] = list(conn.execute("
            f"\"MATCH (n:{table}) RETURN count(n) AS n;\").rows_as_dict())[0]['n']"
        )
    for table in rel_tables:
        lines.append(
            f"counts[{table!r}] = list(conn.execute("
            f"\"MATCH ()-[r:{table}]->() RETURN count(r) AS n;\").rows_as_dict())[0]['n']"
        )
    lines.append("print(json.dumps(counts))")
    return _run_engine(engine, "\n".join(lines))


def _fts_episode_uuids(engine: str, db_path: Path, term: str, fts: Path | None = None) -> list[str]:
    lines = [
        f"db = {engine}.Database({json.dumps(str(db_path))})",
        f"conn = {engine}.Connection(db)",
    ]
    if engine == "ladybug":
        assert fts is not None
        lines.append(f"conn.execute(\"LOAD EXTENSION '{fts}';\")")
        # D-8: the migration strips the definitions; the next initialization
        # recreates them. This child does what `_ensure_fts_indexes` does.
        # Create only if absent: every call after the first reopens the same
        # file, where the index already exists (the same rule _ensure_fts_indexes
        # follows).
        lines.append(
            "existing = {r['index_name'] for r in "
            'conn.execute("CALL SHOW_INDEXES() RETURN *;").rows_as_dict()}'
        )
        lines.append(
            "if 'episode_content' not in existing:\n"
            "    conn.execute(\"CALL CREATE_FTS_INDEX('Episodic','episode_content',"
            "['content','source','source_description']);\")"
        )
    lines.append(
        f"rows = list(conn.execute(\"CALL QUERY_FTS_INDEX('Episodic','episode_content',"
        f"'{term}') RETURN node.uuid AS uuid;\").rows_as_dict())"
    )
    lines.append("print(json.dumps(sorted(r['uuid'] for r in rows)))")
    return _run_engine(engine, "\n".join(lines))


@pytest.fixture
def kuzu_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Kùzu-backed graph with three distinctive episodes and no LLM traffic."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", str(data))
    monkeypatch.delenv("DOCUGRAPH_STORAGE__GRAPH_DB", raising=False)

    import docugraph.storage.graph_store as gs
    from docugraph.core.config import reset_config

    reset_config()
    monkeypatch.setattr(gs, "get_graphiti_llm_client", _fake_llm_client)
    monkeypatch.setattr(gs, "_get_embedder", lambda: gs.GraphitiEmbedderAdapter(FakeEmbedder()))

    async def seed() -> None:
        store = gs.GraphStore()
        try:
            for name, content in (
                ("ep-quasar", "The quasar lattice is stable."),
                ("ep-neutrino", "A neutrino passes through the lattice."),
                ("ep-phonon", "A phonon is a lattice vibration."),
            ):
                await store.add_episode(name=name, content=content, group_id="default")
        finally:
            await store.close()

    import asyncio

    asyncio.run(seed())
    yield data
    reset_config()


def test_p0_5_strip_is_quote_aware_and_removes_only_fts_statements(tmp_path):
    """P0-5: the generated macro carries a `;` inside a quoted regex."""
    schema = (
        "CREATE NODE TABLE T(id STRING, txt STRING, PRIMARY KEY(id));\n"
        "CREATE MACRO 0_T_FTS_TOKENIZE(s) AS regexp_replace(s, '[^a-zA-Z0-9_;]', ' ');\n"
        "CREATE REL TABLE R(FROM T TO T);\n"
    )
    index = "CALL CREATE_FTS_INDEX('T', 't_fts', ['txt']);\n"
    (tmp_path / "schema.cypher").write_text(schema)
    (tmp_path / "index.cypher").write_text(index)

    removed = strip_fts_definitions(tmp_path)

    assert removed == 2
    schema_after = (tmp_path / "schema.cypher").read_text()
    assert "CREATE MACRO" not in schema_after
    assert "CREATE NODE TABLE T" in schema_after
    assert "CREATE REL TABLE R" in schema_after
    assert (tmp_path / "index.cypher").read_text().strip() == ""


def test_m1_migrates_counts_rows_and_searches(kuzu_graph, fts_artifact):
    """M1: counts equal per table, episode text survives, three searches match."""
    from click.testing import CliRunner

    from docugraph.interfaces.cli import main

    kuzu_path = kuzu_graph / "graph" / "graphiti.kuzu"
    lbdb_path = kuzu_graph / "graph" / "graphiti.lbdb"
    before = _counts("kuzu", kuzu_path)
    searches_before = {
        term: _fts_episode_uuids("kuzu", kuzu_path, term)
        for term in ("quasar", "neutrino", "phonon")
    }
    assert all(searches_before.values())

    result = CliRunner().invoke(main, ["graph", "migrate-to-ladybug"])
    assert result.exit_code == 0, result.output
    assert lbdb_path.is_file()
    assert kuzu_path.exists()

    assert _counts("ladybug", lbdb_path, fts_artifact) == before
    assert {
        term: _fts_episode_uuids("ladybug", lbdb_path, term, fts_artifact)
        for term in ("quasar", "neutrino", "phonon")
    } == searches_before

    # The command does not move the setting; the user decides when to switch.
    from docugraph.core.config import Config

    assert Config().storage.graph_db == "kuzu"


def test_m2_an_existing_lbdb_is_never_overwritten(kuzu_graph):
    from click.testing import CliRunner

    from docugraph.interfaces.cli import main

    lbdb_path = kuzu_graph / "graph" / "graphiti.lbdb"
    lbdb_path.write_bytes(b"existing database")

    result = CliRunner().invoke(main, ["graph", "migrate-to-ladybug"])

    assert result.exit_code == 1
    assert lbdb_path.read_bytes() == b"existing database"


def test_m3_a_count_mismatch_removes_the_partial_database(kuzu_graph, monkeypatch):
    from docugraph.storage import graph_migration

    original = graph_migration._run_import

    def wrong_counts(*args: Any, **kwargs: Any) -> dict[str, int]:
        counts = original(*args, **kwargs)
        counts["Episodic"] = counts.get("Episodic", 0) + 1
        return counts

    monkeypatch.setattr(graph_migration, "_run_import", wrong_counts)

    with pytest.raises(MigrationError):
        graph_migration.migrate_graph(kuzu_graph)

    assert not (kuzu_graph / "graph" / "graphiti.lbdb").exists()


def test_the_children_assert_a_single_engine():
    """The guard mutation check: each child refuses to carry the other engine."""
    from docugraph.storage import graph_migration

    assert "ladybug" in graph_migration._EXPORT_SCRIPT
    assert "kuzu" in graph_migration._IMPORT_SCRIPT
    assert "not in sys.modules" in graph_migration._EXPORT_SCRIPT
    assert "not in sys.modules" in graph_migration._IMPORT_SCRIPT
