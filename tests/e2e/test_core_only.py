"""Control rows: what must keep working in a core-only install.

ORACLE: WP-docugraph-0.3.0 §3.B2 (tool list unchanged), §3.B1 (local index and
search need no extra), §5 Part B rows E4 and E6. Both are expected green at the
base: they are controls, not new guards.
"""

from __future__ import annotations

import json
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).parent / "tools_snapshot.json"


def _tool_surface(tool) -> dict:
    annotations = tool.annotations
    return {
        "name": tool.name,
        "inputSchema": tool.inputSchema,
        "annotations": (
            None
            if annotations is None
            else {
                "readOnlyHint": annotations.readOnlyHint,
                "destructiveHint": annotations.destructiveHint,
            }
        ),
    }


def test_e4_tool_list_is_unchanged_without_extras(run_blocked):
    """ORACLE E4: tools/list advertises exactly the frozen surface when the
    extras are absent."""
    proc, payload = run_blocked(
        """
import asyncio

import docugraph.interfaces.mcp_server as server

tools = asyncio.run(server.list_tools())


def surface(tool):
    annotations = tool.annotations
    return {
        "name": tool.name,
        "inputSchema": tool.inputSchema,
        "annotations": (
            None
            if annotations is None
            else {
                "readOnlyHint": annotations.readOnlyHint,
                "destructiveHint": annotations.destructiveHint,
            }
        ),
    }


print(json.dumps(sorted((surface(t) for t in tools), key=lambda t: t["name"])))
"""
    )
    assert proc.returncode == 0, proc.stderr
    assert payload is not None, proc.stderr
    assert payload == json.loads(SNAPSHOT_PATH.read_text())


def test_e6_local_index_and_search_docs_work_without_extras(run_blocked, tmp_path):
    """ORACLE E6: local indexing and the search_docs tool need no extra."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "widget.md").write_text(
        "# Widget Configuration\n\n"
        "To configure a widget, set the sprocket tension to eleven. "
        "The sprocket assembly controls widget alignment and must be "
        "calibrated before the widget is placed into production service."
    )

    proc, payload = run_blocked(
        f"""
import asyncio

from click.testing import CliRunner

import docugraph.interfaces.mcp_server as server
from docugraph.interfaces.cli import main

indexed = CliRunner().invoke(main, ["index", "local", {str(docs)!r}])
searched = asyncio.run(
    server._search_docs({{"query": "how do I configure a widget"}})
)
print(json.dumps({{
    "index_exit": indexed.exit_code,
    "search_is_error": bool(searched.isError),
    "text": "\\n".join(
        item.text for item in searched.content if getattr(item, "text", None)
    ),
}}))
"""
    )
    assert proc.returncode == 0, proc.stderr
    assert payload is not None, proc.stderr
    assert payload["index_exit"] == 0
    assert payload["search_is_error"] is False
    assert "sprocket" in payload["text"].lower()
