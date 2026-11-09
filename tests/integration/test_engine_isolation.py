"""Q-1: with the default Kùzu backend, Ladybug never enters the process.

The two engines cannot share a process, so the production paths that touch the
graph -- an MCP graph tool, hybrid search's graph leg, and the CLI's graph
commands -- must load only the configured engine. The child process below runs
all of them with ``ladybug`` blocked at import; if any path tried to load it,
the import would raise, and ``sys.modules`` is checked as well.

``fetch-fts`` and ``install-fts`` are not part of that rule: they are their own
CLI invocations, one process each, and no MCP tool or library path reaches
them. The second test pins that structurally.

ORACLE: PR #28 round-1 review, Q-1 (production engines never mix).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_CHILD = """
import asyncio
import json
import sys

from click.testing import CliRunner

from docugraph.interfaces.cli import main

report = {}

search = CliRunner().invoke(main, ["graph", "search", "anything"])
report["cli_graph_search_exit"] = search.exit_code

clear = CliRunner().invoke(main, ["graph", "clear"], input="y\\n")
report["cli_graph_clear_exit"] = clear.exit_code

from docugraph.interfaces import mcp_server

tool_result = asyncio.run(mcp_server.call_tool("graph_query", {"query": "anything"}))
report["mcp_graph_query_is_error"] = bool(tool_result.isError)
report["tool_names"] = sorted(tool.name for tool in asyncio.run(mcp_server.list_tools()))

from docugraph.retrieval.hybrid_search import HybridRetriever, SearchMode


class FakeVectorStore:
    def search(self, query, top_k=10, filter_expr=None):
        return []

    def search_fts(self, query, top_k=10, filter_expr=None):
        return []


retriever = HybridRetriever(vector_store=FakeVectorStore())
retriever.search("anything", mode=SearchMode.ALL)
report["hybrid_graph_available"] = retriever.graph_status.available
report["hybrid_reason_is_a_sentence"] = bool(retriever.graph_status.reason) or (
    retriever.graph_status.available
)

report["ladybug_in_sys_modules"] = "ladybug" in sys.modules
report["lbug_in_sys_modules"] = "ladybug._lbug" in sys.modules
print(json.dumps(report))
"""


def test_q1_the_kuzu_backend_never_loads_ladybug(run_blocked):
    """Every production graph path runs with the Ladybug package unimportable."""
    proc, payload = run_blocked(_CHILD, roots=("ladybug",))

    assert proc.returncode == 0, proc.stderr
    assert payload is not None, proc.stdout
    assert payload["ladybug_in_sys_modules"] is False
    assert payload["lbug_in_sys_modules"] is False
    # The graph commands ran; they may report an unavailable LLM (exit 3) or an
    # empty graph, but they must not have needed Ladybug to do it.
    assert payload["cli_graph_search_exit"] in (0, 3)
    assert payload["cli_graph_clear_exit"] == 0


def test_q1_no_mcp_tool_or_library_path_installs_or_fetches_the_extension(run_blocked):
    """The unverified install is a CLI command, never a tool and never a library call."""
    proc, payload = run_blocked(_CHILD, roots=("ladybug",))
    assert proc.returncode == 0, proc.stderr
    assert payload is not None, proc.stdout

    names = payload["tool_names"]
    assert "graph_query" in names
    assert "graph_add" in names
    assert "fetch-fts" not in names
    assert "install-fts" not in names

    # One executed `INSTALL FTS;` statement exists in the package, inside the
    # function the `graph install-fts --unverified` command alone calls.
    hits = [
        f"{path.relative_to(REPO_ROOT)}:{number}"
        for path in sorted((REPO_ROOT / "src" / "docugraph").rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if '"INSTALL FTS;"' in line
    ]
    assert len(hits) == 1, hits
    assert hits[0].startswith("src/docugraph/storage/fts.py:")

    from docugraph.storage import fts

    source = Path(fts.__file__).read_text()
    assert "def install_unverified" in source
