"""Shared test configuration.

Two jobs:

* register the ``requires_extras`` marker and apply it by inspection, so the
  same test files run in the full profile and are deselected in the minimal
  one (``-m "not requires_extras"``). It is registered here rather than in
  ``pyproject.toml`` so the test suite owns its own vocabulary.
* provide the ``run_blocked`` fixture: a child process that simulates a
  core-only install by refusing the optional dependency roots. These tests
  therefore prove the minimal install's behaviour in *both* profiles.
* keep the two graph engines apart. Kùzu and Ladybug bind the same C++ type
  names, and once one is imported the other's extension cannot load, so a
  process can run only one of them. Modules that drive Ladybug carry the
  ``ladybug_engine`` marker; a mixed collection deselects them and says so, and
  ``--ladybug-only`` runs just them (the CI job ``Graph (Ladybug)``).

The ``requires_extras`` marker is applied in ``pytest_collection_modifyitems``.
Conftest plugins are registered after third-party plugins, so for this
non-wrapper hook they run first — before the mark plugin applies ``-m``
deselection.
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
    "test_engine_isolation.py": "graph",
    "test_ladybug_driver.py": "ladybug",
    "test_graph_ladybug.py": "ladybug",
    "test_graph_migration.py": "ladybug",
    "test_fts_manifest.py": "ladybug",
}

_EXTRA_PROBE = {"crawl": "crawl4ai", "graph": "graphiti_core", "ladybug": "ladybug"}

# Kùzu and Ladybug bind the same C++ type names. Constructing a database with
# one makes the other's extension unimportable for the rest of the process
# (`ImportError: generic_type: type "Database" is already registered!`), and
# nothing can undo that. Tests that construct Ladybug databases therefore carry
# this marker and are run last: every module that needs Kùzu has then already
# imported and used it. The migration command lives behind the same constraint
# and does its work in child processes, one engine each.
#
# The marker is applied with `pytestmark` in the affected modules; the hook
# below is what enforces the ordering, for any selection of tests.
_ENGINE_MARKER = "ladybug_engine"

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
    config.addinivalue_line(
        "markers",
        f"{_ENGINE_MARKER}: constructs a Ladybug database in this process; "
        "ordered after every Kùzu test (the two engines cannot share a process)",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--ladybug-only",
        action="store_true",
        help=(
            "run only the LadybugDB engine tests (the default run deselects "
            "them, because the two engines cannot share a process)"
        ),
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        module = item.nodeid.split("::", 1)[0].rsplit("/", 1)[-1]
        extra = _REQUIRED_EXTRA.get(module)
        if extra is None:
            continue
        if importlib.util.find_spec(_EXTRA_PROBE[extra]) is None:
            item.add_marker(pytest.mark.requires_extras)

    ladybug = [item for item in items if item.get_closest_marker(_ENGINE_MARKER)]
    kuzu = [item for item in items if item.get_closest_marker(_ENGINE_MARKER) is None]

    # A mixed collection is impossible to run: importing either engine makes
    # the other's extension unimportable, so whichever ran first would make
    # every later test fail. Deselect the Ladybug group when both are present,
    # and say so at the end of the run -- the CI job `Graph (Ladybug)` is a
    # separate invocation that runs exactly this group.
    if ladybug and kuzu and not config.getoption("--ladybug-only"):
        items[:] = kuzu
        config.hook.pytest_deselected(items=ladybug)
        config._docugraph_ladybug_deselected = len(ladybug)  # type: ignore[attr-defined]
    elif config.getoption("--ladybug-only"):
        items[:] = ladybug
        config.hook.pytest_deselected(items=kuzu)


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, exitstatus: int, config: pytest.Config
) -> None:
    deselected = getattr(config, "_docugraph_ladybug_deselected", 0)
    if not deselected:
        return
    terminalreporter.write_sep("-", "LadybugDB tests not run in this invocation")
    terminalreporter.write_line(
        f"{deselected} test(s) drive LadybugDB. Kùzu and Ladybug bind the same C++ "
        "type names, so one process can import only one of them; run the other "
        "group in a second invocation:"
    )
    terminalreporter.write_line(
        "  pytest --ladybug-only tests/integration/test_ladybug_driver.py "
        "tests/integration/test_graph_ladybug.py tests/unit/test_fts_manifest.py"
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Catch a module that activates Ladybug without the engine marker."""
    if item.get_closest_marker(_ENGINE_MARKER) is not None:
        return
    if "ladybug._lbug" in sys.modules:
        pytest.fail(
            f"{item.nodeid} runs in a process where LadybugDB has already been "
            "activated, so it cannot import Kùzu. If this module drives a "
            f"Ladybug database, mark it with pytest.mark.{_ENGINE_MARKER}; the "
            "two engines cannot share a process.",
            pytrace=False,
        )


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
        script.write_text(
            _BLOCKER_TEMPLATE.format(roots=list(roots)) + "\nimport json\nimport os\n\n" + body
        )
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
