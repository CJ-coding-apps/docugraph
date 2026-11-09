"""The Ladybug backend end to end: fail-closed FTS, D-3, no installer.

ORACLE: WP-docugraph-0.4.0 §3.1/§3.2, §5 L3–L8, §10.2 D-5/D-6/D-8/D-9,
§10.6 F2/F5, §10.9 note 1 (FTS is database-scoped; the pool must inherit it).

Every test that needs a real FTS artifact uses the session ``fts_artifact``
fixture (a pinned, verified file). Without one the FTS-dependent tests skip
with the provisioning command in the reason; CI fetches it first.
"""

from __future__ import annotations

import re
import shutil
import socket
from pathlib import Path
from typing import Any

import pytest

from backend_oracle import (
    FTS_INIT_FAILED,
    FTS_NO_FILE,
    FTS_NO_PIN,
    GRAPH_NEEDS_MIGRATION_MESSAGE,
)
from docugraph.core.config import reset_config

# Some of these tests construct a Ladybug database in-process, which makes
# Kùzu's extension unimportable here; tests/conftest.py runs this module last.
pytestmark = pytest.mark.ladybug_engine


class FakeEmbedder:
    """A fixed vector, so graph ingestion needs no ONNX model download."""

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


def _make_fake_llm_client() -> Any:
    """Graphiti LLM client that extracts nothing and never reaches the network."""
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


@pytest.fixture
def ladybug_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", str(data))
    monkeypatch.setenv("DOCUGRAPH_STORAGE__GRAPH_DB", "ladybug")
    reset_config()
    yield data
    reset_config()


@pytest.fixture
def offline_graph_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    import docugraph.storage.graph_store as gs

    monkeypatch.setattr(gs, "get_graphiti_llm_client", _make_fake_llm_client)
    monkeypatch.setattr(gs, "_get_embedder", lambda: gs.GraphitiEmbedderAdapter(FakeEmbedder()))


def _seed_verified_fts(data_dir: Path, artifact: Path) -> Path:
    from docugraph.storage import fts

    target = fts.cache_path(data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifact, target)
    return target


def _no_artifact_message() -> str:
    from docugraph.storage import fts, fts_pins

    pin = fts_pins.PINS.get((fts.compat_version(), fts.platform_key()))
    return FTS_NO_FILE if pin is not None else FTS_NO_PIN


# --------------------------------------------------------------------- L3


def test_l3_full_cycle_on_ladybug_with_kuzu_unimportable(ladybug_data, fts_artifact, run_blocked):
    """L3: add and search work; `kuzu` and its graphiti driver never load."""
    proc, payload = run_blocked(
        """
import asyncio
import shutil
from pathlib import Path

from docugraph.storage import fts

data = Path(os.environ["DOCUGRAPH_STORAGE__DATA_DIR"])
artifact = Path(os.environ["DOCUGRAPH_TEST_FTS_ARTIFACT"])
target = fts.cache_path(data)
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(artifact, target)

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig


class FakeLLM(LLMClient):
    def __init__(self):
        super().__init__(config=LLMConfig(model="fake", small_model="fake"))

    async def _generate_response(self, messages, response_model=None,
                                 max_tokens=None, model_size=None):
        if response_model is None:
            return {}
        return {
            name: []
            for name, field in response_model.model_fields.items()
            if field.annotation is not None and "list" in str(field.annotation).lower()
        }


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, query):
        return [0.1] * 8

    @property
    def dimensions(self):
        return 8

    @property
    def model_name(self):
        return "fake"


import docugraph.storage.graph_store as gs

gs.get_graphiti_llm_client = FakeLLM
gs._get_embedder = lambda: gs.GraphitiEmbedderAdapter(FakeEmbedder())


async def main():
    store = gs.GraphStore()
    try:
        added = await store.add_episode(
            name="ep", content="The quasar lattice is stable.", group_id="default"
        )
        found = await store.search("quasar")
    finally:
        await store.close()
    return added, found


added, found = asyncio.run(main())
print(json.dumps({
    "episode_uuid": added["episode_uuid"],
    "search_is_list": isinstance(found, list),
    "lbdb": (data / "graph" / "graphiti.lbdb").exists(),
    "kuzu_file": (data / "graph" / "graphiti.kuzu").exists(),
    "kuzu_imported": "kuzu" in sys.modules,
    "kuzu_driver_imported": "graphiti_core.driver.kuzu_driver" in sys.modules,
}))
""",
        roots=("kuzu",),
        env_extra={
            "DOCUGRAPH_STORAGE__GRAPH_DB": "ladybug",
            "DOCUGRAPH_TEST_FTS_ARTIFACT": str(fts_artifact),
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert payload is not None, proc.stderr
    assert payload["search_is_list"] is True
    assert payload["lbdb"] is True
    assert payload["kuzu_file"] is False
    assert payload["kuzu_imported"] is False
    assert payload["kuzu_driver_imported"] is False


# --------------------------------------------------------------------- L4


async def test_l4_ladybug_creates_only_the_lbdb_file(
    ladybug_data, fts_artifact, offline_graph_clients
):
    _seed_verified_fts(ladybug_data, fts_artifact)

    import docugraph.storage.graph_store as gs

    store = gs.GraphStore()
    try:
        await store._ensure_initialized()
    finally:
        await store.close()

    graph_dir = ladybug_data / "graph"
    assert (graph_dir / "graphiti.lbdb").is_file()
    assert not (graph_dir / "graphiti.kuzu").exists()


# ------------------------------------------------------------- L5 / F2 MCP


async def test_l5_unmigrated_graph_refuses_before_opening_anything(ladybug_data):
    """L5: the sentence is exact and no `.lbdb` appears beside the old graph."""
    from docugraph.interfaces.mcp_server import _graph_query

    (ladybug_data / "graph" / "graphiti.kuzu").mkdir(parents=True)

    result = await _graph_query({"query": "anything"})

    assert result.isError is True
    assert result.content[0].text == GRAPH_NEEDS_MIGRATION_MESSAGE
    assert not (ladybug_data / "graph" / "graphiti.lbdb").exists()


async def test_f2_no_verified_fts_refuses_before_the_llm_or_the_database(ladybug_data):
    """F2: no artifact means the D-5 sentence, and no database is created.

    The check must precede LLM resolution: with no provider configured this
    environment would otherwise fail with the LLM sentence instead.
    """
    from docugraph.interfaces.mcp_server import _graph_query

    (ladybug_data / "graph").mkdir(parents=True)

    result = await _graph_query({"query": "anything"})

    assert result.isError is True
    assert result.content[0].text == _no_artifact_message()
    assert not (ladybug_data / "graph" / "graphiti.lbdb").exists()


# --------------------------------------------------------------------- L6


def test_l6_cli_reports_the_migration_sentence(ladybug_data, monkeypatch):
    from click.testing import CliRunner

    from docugraph.interfaces.cli import main

    (ladybug_data / "graph" / "graphiti.kuzu").mkdir(parents=True)
    result = CliRunner().invoke(main, ["graph", "search", "anything"])

    output = result.output + (getattr(result, "stderr", "") or "")
    assert result.exit_code == 1
    assert GRAPH_NEEDS_MIGRATION_MESSAGE in output
    assert "Traceback" not in output


def test_f2_cli_reports_the_missing_fts_sentence(ladybug_data):
    from click.testing import CliRunner

    from docugraph.interfaces.cli import main

    (ladybug_data / "graph").mkdir(parents=True)
    result = CliRunner().invoke(main, ["graph", "search", "anything"])

    output = result.output + (getattr(result, "stderr", "") or "")
    assert result.exit_code == 1
    assert _no_artifact_message() in output
    assert "Traceback" not in output
    assert not (ladybug_data / "graph" / "graphiti.lbdb").exists()


# --------------------------------------------------------------------- L7


async def test_l7_no_network_and_only_a_verified_load(
    ladybug_data, fts_artifact, offline_graph_clients, monkeypatch
):
    """L7: no socket, no INSTALL, one LOAD of the verified path."""
    import ladybug

    statement_log: list[str] = []
    original_execute = ladybug.Connection.execute

    def spying_execute(self, query, *args, **kwargs):
        statement_log.append(query)
        return original_execute(self, query, *args, **kwargs)

    monkeypatch.setattr(ladybug.Connection, "execute", spying_execute)

    def no_socket(*args, **kwargs):
        raise AssertionError("the Ladybug backend opened a socket")

    monkeypatch.setattr(socket.socket, "__init__", no_socket)

    _seed_verified_fts(ladybug_data, fts_artifact)

    import docugraph.storage.graph_store as gs

    store = gs.GraphStore()
    try:
        await store.add_episode(name="ep", content="quasar lattice", group_id="default")
        await store.search("quasar")
    finally:
        await store.close()

    installs = [q for q in statement_log if re.match(r"(?i)^\s*INSTALL\b", q)]
    loads = [q for q in statement_log if re.match(r"(?i)^\s*LOAD\b", q)]

    # The store loads the verified copy it resolved from its own data dir; the
    # fixture seeded exactly those bytes at that path. (The blind suite compared
    # against the fixture's source path, which the store never sees.)
    from docugraph.storage import fts

    loaded_path = fts.cache_path(ladybug_data)

    assert installs == []
    assert loads == [f"LOAD EXTENSION '{loaded_path}'"]


# --------------------------------------------------------------------- L8


def test_l8_without_the_ladybug_extra_the_sentence_names_it(run_blocked):
    proc, payload = run_blocked(
        """
import asyncio

from docugraph.interfaces.mcp_server import _graph_query

result = asyncio.run(_graph_query({"query": "anything"}))
print(json.dumps({
    "is_error": bool(result.isError),
    "text": result.content[0].text,
}))
""",
        roots=("ladybug",),
        env_extra={"DOCUGRAPH_STORAGE__GRAPH_DB": "ladybug"},
    )
    assert proc.returncode == 0, proc.stderr
    assert payload == {
        "is_error": True,
        "text": "This feature needs the 'ladybug' extra: pip install 'docugraph[ladybug]'",
    }


# ---------------------------------------------------------- L9 (D-9) / L10


async def test_l9_a_failed_create_is_an_initialization_error(
    ladybug_data, fts_artifact, offline_graph_clients, monkeypatch
):
    """D-9: on Ladybug an FTS create failure stops the backend and names the fix."""
    _seed_verified_fts(ladybug_data, fts_artifact)

    from docugraph.storage.fts import FtsInitializationError
    from docugraph.storage.ladybug_driver import LadybugDriver

    original = LadybugDriver.execute_query

    async def failing(self, query, **kwargs):
        if "CREATE_FTS_INDEX" in query:
            raise RuntimeError("index creation refused by test")
        return await original(self, query, **kwargs)

    monkeypatch.setattr(LadybugDriver, "execute_query", failing)

    import docugraph.storage.graph_store as gs

    store = gs.GraphStore()
    with pytest.raises(FtsInitializationError) as excinfo:
        await store._ensure_initialized()
    await store.close()

    # MED-2: the approved sentence, character for character. It states that
    # the graph was not opened and names the Kùzu fallback.
    assert str(excinfo.value) == FTS_INIT_FAILED.format(error="index creation refused by test")


async def test_l10_every_pooled_connection_can_query_fts(tmp_path, fts_artifact):
    """§10.9 note 1: the load is database-scoped, so the pool inherits it.

    Loading on the setup connection before the pool is constructed is the only
    order that makes every pooled connection able to run the FTS function.
    """
    import asyncio

    from docugraph.storage.ladybug_driver import LadybugDriver

    driver = LadybugDriver(
        db=str(tmp_path / "pool.lbdb"),
        max_concurrent_queries=4,
        fts_extension=str(fts_artifact),
    )
    try:
        await driver.execute_query("CREATE NODE TABLE Doc(id INT64, txt STRING, PRIMARY KEY(id));")
        await driver.execute_query("CREATE (:Doc {id: 1, txt: 'quasar lattice'});")
        await driver.execute_query("CREATE (:Doc {id: 2, txt: 'lattice phonon'});")
        await driver.execute_query("CALL CREATE_FTS_INDEX('Doc','doc_fts',['txt']);")

        pool = driver.client.connections
        assert len(pool) == 4
        for connection in pool:
            rows = list(
                connection.execute(
                    "CALL QUERY_FTS_INDEX('Doc','doc_fts','quasar') RETURN node.id AS id;"
                ).rows_as_dict()
            )
            assert [row["id"] for row in rows] == [1]

        # The behaviour the structural check stands for: concurrent queries
        # through the public path all reach a connection that can run FTS.
        results = await asyncio.gather(
            *[
                driver.execute_query(
                    "CALL QUERY_FTS_INDEX('Doc','doc_fts','lattice') "
                    "RETURN node.id AS id ORDER BY id;"
                )
                for _ in range(4)
            ]
        )
        assert all(rows == [{"id": 1}, {"id": 2}] for rows, _, _ in results)
    finally:
        await driver.close()


# --------------------------------------------------------------------- F5


async def test_f5_add_and_bm25_search_return_the_seeded_fact(
    ladybug_data, fts_artifact, offline_graph_clients
):
    """F5: the BM25 leg docugraph reaches works on Ladybug after `add`."""
    _seed_verified_fts(ladybug_data, fts_artifact)

    import docugraph.storage.graph_store as gs

    store = gs.GraphStore()
    try:
        await store.add_episode(
            name="ep", content="The quasar lattice is stable.", group_id="default"
        )
        driver = store._driver
        await driver.execute_query(
            "CREATE (:RelatesToNode_ {uuid: 'edge-1', name: 'quasar relation', "
            "fact: 'the quasar lattice is stable', episodes: []});"
        )
        rows, _, _ = await driver.execute_query(
            "CALL QUERY_FTS_INDEX('RelatesToNode_','edge_name_and_fact','quasar') "
            "RETURN node.uuid AS uuid;"
        )
        episodes, _, _ = await driver.execute_query(
            "CALL QUERY_FTS_INDEX('Episodic','episode_content','quasar') RETURN node.uuid AS uuid;"
        )
    finally:
        await store.close()

    assert rows == [{"uuid": "edge-1"}]
    assert len(episodes) == 1
