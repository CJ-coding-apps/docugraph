"""Missing-extra behaviour: one exact sentence, in every surface.

Each test runs a child process in which the optional dependency roots cannot be
imported (``run_blocked`` in ``tests/conftest.py``), i.e. a core-only install.

ORACLE: WP-docugraph-0.3.0 §3.B2, §5 Part B rows E1, E2, E2b, E3, E5, E5b, E7,
A1t, and §6.
"""

from __future__ import annotations

import pytest

CRAWL_MESSAGE = (
    "This feature needs the 'crawl' extra: pip install 'docugraph[crawl]', "
    "then: playwright install chromium"
)
GRAPH_MESSAGE = "This feature needs the 'graph' extra: pip install 'docugraph[graph]'"

LAZY_PUBLIC_NAMES = (
    "DocCrawler",
    "Chunker",
    "VectorStore",
    "HybridRetriever",
    "SearchMode",
    "FusionStrategy",
    "UrlPolicy",
    "MissingExtraError",
)


# ---------------------------------------------------------------- E1 / A1t


def test_e1_import_docugraph_does_not_import_the_extras(run_blocked):
    """ORACLE E1/B4: importing the package and every declared public name
    succeeds, and none of crawl4ai, graphiti_core, kuzu or playwright is
    imported."""
    proc, payload = run_blocked(
        f"""
import docugraph

names = {list(LAZY_PUBLIC_NAMES)!r}
missing = [name for name in names if not hasattr(docugraph, name)]
print(json.dumps({{
    "missing": missing,
    "blocked_present": sorted(
        root for root in {["crawl4ai", "graphiti_core", "kuzu", "playwright"]!r}
        if root in sys.modules
    ),
}}))
"""
    )
    assert proc.returncode == 0, proc.stderr
    assert payload is not None, proc.stderr
    assert payload["missing"] == []
    assert payload["blocked_present"] == []


def test_a1t_public_names_construct_and_crawl_raises_missing_extra(run_blocked):
    """ORACLE A1t: constructing DocCrawler is fine; crawling raises
    MissingExtraError carrying the exact crawl sentence."""
    proc, payload = run_blocked(
        """
import asyncio

from docugraph import DocCrawler, MissingExtraError

assert issubclass(MissingExtraError, ImportError)
crawler = DocCrawler()
try:
    asyncio.run(crawler.crawl_single("https://example.com"))
except MissingExtraError as exc:
    print(json.dumps({"type": type(exc).__name__, "text": str(exc)}))
else:
    print(json.dumps({"type": None, "text": None}))
"""
    )
    assert proc.returncode == 0, proc.stderr
    assert payload == {"type": "MissingExtraError", "text": CRAWL_MESSAGE}


# ---------------------------------------------------------------- E2 / E2b / E3


def test_e2_crawl_url_core_only_returns_the_exact_crawl_message(run_blocked):
    """ORACLE E2: MCP crawl_url returns isError with exactly the crawl
    sentence, and nothing prints a traceback to stderr."""
    proc, payload = run_blocked(
        """
import asyncio

import docugraph.interfaces.mcp_server as server
from docugraph.security.url_policy import UrlKind, UrlPolicy

server._web_policy = lambda: UrlPolicy(
    kind=UrlKind.WEB, resolver=lambda host: ("93.184.216.34",)
)
result = asyncio.run(server._crawl_url({"url": "https://example.com"}))
print(json.dumps({
    "isError": bool(result.isError),
    "text": "\\n".join(
        item.text for item in result.content if getattr(item, "text", None)
    ),
}))
"""
    )
    assert proc.returncode == 0, proc.stderr
    assert payload == {"isError": True, "text": CRAWL_MESSAGE}
    assert "Traceback" not in proc.stderr


def test_e2b_call_tool_backstop_formats_a_handler_without_a_clause(run_blocked):
    """ORACLE E2b: graph_add carries no handler clause; the call_tool backstop
    returns the exact graph sentence, and its log line carries no traceback."""
    proc, payload = run_blocked(
        """
import asyncio

import docugraph.interfaces.mcp_server as server

result = asyncio.run(server.call_tool("graph_add", {"content": "x"}))
print(json.dumps({
    "isError": bool(result.isError),
    "text": "\\n".join(
        item.text for item in result.content if getattr(item, "text", None)
    ),
}))
"""
    )
    assert proc.returncode == 0, proc.stderr
    assert payload == {"isError": True, "text": GRAPH_MESSAGE}
    assert "Traceback" not in proc.stderr


def test_e3_graph_query_core_only_returns_the_exact_graph_message(run_blocked):
    """ORACLE E3: MCP graph_query returns isError with exactly the graph
    sentence."""
    proc, payload = run_blocked(
        """
import asyncio

import docugraph.interfaces.mcp_server as server

result = asyncio.run(server._graph_query({"query": "x"}))
print(json.dumps({
    "isError": bool(result.isError),
    "text": "\\n".join(
        item.text for item in result.content if getattr(item, "text", None)
    ),
}))
"""
    )
    assert proc.returncode == 0, proc.stderr
    assert payload == {"isError": True, "text": GRAPH_MESSAGE}


# ---------------------------------------------------------------- E5 / E5b


def test_e5_cli_crawl_core_only_exits_non_zero_with_the_exact_message(run_blocked):
    """ORACLE E5: the CLI exits 1 and prints exactly the crawl sentence — no
    traceback and no extra prefix."""
    proc, payload = run_blocked(
        """
from click.testing import CliRunner

from docugraph.interfaces.cli import main

result = CliRunner().invoke(main, ["crawl", "https://example.com"])
print(json.dumps({
    "exit_code": result.exit_code,
    "stdout": result.stdout,
    "stderr": getattr(result, "stderr", ""),
    "exception": None if result.exception is None else type(result.exception).__name__,
}))
"""
    )
    assert proc.returncode == 0, proc.stderr
    assert payload is not None, proc.stderr
    combined = payload["stdout"] + payload["stderr"]
    assert payload["exit_code"] == 1
    assert CRAWL_MESSAGE in combined
    assert "Traceback" not in combined
    assert "Error:" not in combined
    assert payload["exception"] in (None, "SystemExit")


def test_e5b_cli_help_works_core_only(run_blocked):
    """ORACLE E5b/B2: `docugraph --help` exits 0 with core only."""
    proc, payload = run_blocked(
        """
import sys

from click.testing import CliRunner

from docugraph.interfaces.cli import main

result = CliRunner().invoke(main, ["--help"])
print(json.dumps({
    "exit_code": result.exit_code,
    "blocked_present": sorted(
        root for root in ("crawl4ai", "graphiti_core", "kuzu", "playwright")
        if root in sys.modules
    ),
}))
"""
    )
    assert proc.returncode == 0, proc.stderr
    assert payload == {"exit_code": 0, "blocked_present": []}


# ---------------------------------------------------------------- E7


def test_e7_policy_runs_before_the_extra_check(run_blocked):
    """ORACLE E7: a refused URL keeps its refusal in a core-only install — the
    policy decision does not depend on what is installed."""
    proc, payload = run_blocked(
        """
import asyncio

import docugraph.interfaces.mcp_server as server

result = asyncio.run(server._crawl_url({"url": "file:///etc/hosts"}))
print(json.dumps({
    "isError": bool(result.isError),
    "text": "\\n".join(
        item.text for item in result.content if getattr(item, "text", None)
    ),
}))
"""
    )
    assert proc.returncode == 0, proc.stderr
    assert payload is not None, proc.stderr
    assert payload["isError"] is True
    assert payload["text"].startswith("Refused: scheme_not_allowed")


# ---------------------------------------------------------------- the message itself


def test_crawl_message_is_exact_and_names_only_the_crawl_extra():
    """ORACLE B2: one sentence, from one function, with the browser step."""
    from docugraph._optional import MissingExtraError, require

    with pytest.raises(MissingExtraError) as info:
        require("crawl", "docugraph_test_no_such_module")
    assert str(info.value) == CRAWL_MESSAGE
    assert isinstance(info.value, ImportError)
    assert "graph" not in str(info.value)


def test_graph_message_is_exact_and_names_only_the_graph_extra():
    """ORACLE B2: the graph sentence, and no crawl instruction."""
    from docugraph._optional import MissingExtraError, require

    with pytest.raises(MissingExtraError) as info:
        require("graph", "docugraph_test_no_such_module")
    assert str(info.value) == GRAPH_MESSAGE
    assert isinstance(info.value, ImportError)
    assert "crawl" not in str(info.value)


# ---------------------------------------------------------------- independence (§6)


def test_crawl_and_graph_fail_independently(run_blocked):
    """ORACLE §6: with only graph absent, the crawler and vector store import;
    with only crawl absent, the retriever and graph store module import."""
    proc, payload = run_blocked(
        """
from docugraph import DocCrawler, VectorStore

print(json.dumps({"crawl_ok": DocCrawler is not None and VectorStore is not None}))
""",
        roots=("graphiti_core", "kuzu"),
    )
    assert proc.returncode == 0, proc.stderr
    assert payload == {"crawl_ok": True}

    proc, payload = run_blocked(
        """
import importlib

from docugraph import HybridRetriever

graph_store = importlib.import_module("docugraph.storage.graph_store")
print(json.dumps({"graph_ok": HybridRetriever is not None and graph_store is not None}))
""",
        roots=("crawl4ai", "playwright"),
    )
    assert proc.returncode == 0, proc.stderr
    assert payload == {"graph_ok": True}
