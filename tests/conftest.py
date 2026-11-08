"""Shared test configuration.

Two jobs:

* register the ``requires_extras`` marker and apply it by inspection, so the
  same test files run in the full profile and are deselected in the minimal
  one (``-m "not requires_extras"``). It is registered here rather than in
  ``pyproject.toml`` so the test suite owns its own vocabulary.
* provide the ``run_blocked`` fixture: a child process that simulates a
  core-only install by refusing the optional dependency roots. These tests
  therefore prove the minimal install's behaviour in *both* profiles.

The marker is applied in ``pytest_collection_modifyitems``. Conftest plugins
are registered after third-party plugins, so for this non-wrapper hook they run
first — before the mark plugin applies ``-m`` deselection.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# The roots an extra installs. Blocking a root makes every submodule import
# fail exactly as it would in an install without that extra.
BLOCKED_ROOTS = ("crawl4ai", "graphiti_core", "kuzu", "playwright")

# Tests that cannot run without an extra, by module basename. Any module not
# listed here runs in both profiles.
_REQUIRED_EXTRA = {
    "test_crawl_guard.py": "crawl",
    "test_crawl_antibot.py": "crawl",
    "test_crawl_policy.py": "crawl",
    "test_crawl_redirect.py": "crawl",
    "test_browser_require.py": "crawl",
    "test_event_loop_acquisition.py": "graph",
    "test_graphiti_llm_client.py": "graph",
    "test_hybrid_graph_status.py": "graph",
    "test_llm_preflight.py": "graph",
    "test_telemetry_default.py": "graph",
    "test_graph_store_init.py": "graph",
    "test_mcp_server.py": "graph",
    "test_graph_embedder_adapter.py": "graph",
}

_EXTRA_PROBE = {"crawl": "crawl4ai", "graph": "graphiti_core"}

_BLOCKER_TEMPLATE = '''\
import sys


class _Blocker:
    """Refuse the given root packages, as a core-only install would."""

    def __init__(self, roots):
        self.roots = frozenset(roots)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.roots:
            raise ModuleNotFoundError(
                f"No module named {{fullname!r}}", name=fullname
            )
        return None


sys.meta_path.insert(0, _Blocker({roots!r}))
'''


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_extras: needs an optional dependency group; deselected in "
        'the minimal profile with -m "not requires_extras"',
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        module = item.nodeid.split("::", 1)[0].rsplit("/", 1)[-1]
        extra = _REQUIRED_EXTRA.get(module)
        if extra is None:
            continue
        if importlib.util.find_spec(_EXTRA_PROBE[extra]) is None:
            item.add_marker(pytest.mark.requires_extras)


@pytest.fixture
def run_blocked(tmp_path: Path):
    """Run ``body`` in a child process that cannot import the extra roots.

    The child prints one JSON line; the fixture returns the completed process
    plus the parsed payload (``None`` when the child printed none). A temp data
    directory is exported so no test writes into the real one.
    """

    def _run(
        body: str,
        *,
        roots: tuple[str, ...] = BLOCKED_ROOTS,
        env_extra: dict[str, str] | None = None,
        timeout: int = 600,
    ) -> tuple[subprocess.CompletedProcess[str], dict | None]:
        script = tmp_path / "blocked_child.py"
        script.write_text(_BLOCKER_TEMPLATE.format(roots=list(roots)) + "\nimport json\n\n" + body)
        env = os.environ.copy()
        env["DOCUGRAPH_STORAGE__DATA_DIR"] = str(tmp_path / "data")
        env.update(env_extra or {})
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(tmp_path),
            timeout=timeout,
        )
        payload = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            break
        return proc, payload

    return _run
