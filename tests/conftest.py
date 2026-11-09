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
BLOCKED_ROOTS = ("crawl4ai", "graphiti_core", "kuzu", "ladybug", "playwright")

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
    "test_ladybug_driver.py": "ladybug",
    "test_graph_ladybug.py": "ladybug",
    "test_graph_migration.py": "ladybug",
    "test_fts_manifest.py": "ladybug",
}

_EXTRA_PROBE = {"crawl": "crawl4ai", "graph": "graphiti_core", "ladybug": "ladybug"}

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


@pytest.fixture(scope="session")
def fts_artifact() -> Path:
    """A verified FTS artifact for the running platform, or a skip.

    ``DOCUGRAPH_TEST_FTS_ARTIFACT`` points at a file already downloaded (the
    Phase 0 measurements live in /tmp, not in the repo). Otherwise the cache is
    consulted, and only ``DOCUGRAPH_TEST_FETCH_FTS=1`` allows a download. CI's
    Ladybug job fetches once into the cache before the test run.
    """
    override = os.environ.get("DOCUGRAPH_TEST_FTS_ARTIFACT")
    if override:
        return Path(override)

    import tempfile

    from docugraph.storage import fts

    with tempfile.TemporaryDirectory() as scratch:
        try:
            return fts.resolve_verified_artifact(Path(scratch))
        except Exception:  # noqa: BLE001 - any absence means "not provisioned"
            pass

    if os.environ.get("DOCUGRAPH_TEST_FETCH_FTS") == "1":
        from docugraph.core.config import get_config
        from docugraph.storage import fts

        return fts.fetch_fts(get_config().storage.data_dir)

    pytest.skip(
        "no verified FTS artifact: set DOCUGRAPH_TEST_FTS_ARTIFACT=<file>, "
        "or DOCUGRAPH_TEST_FETCH_FTS=1 to download it"
    )


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
