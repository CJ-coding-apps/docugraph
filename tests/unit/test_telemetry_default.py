"""Graphiti telemetry must be off unless the user opted in.

graphiti-core reads ``GRAPHITI_TELEMETRY_ENABLED`` at construction time and
treats an absent variable as enabled. The suite never saw it because graphiti
also disables itself when ``pytest`` is in ``sys.modules`` -- which is why T1
runs a child process that never imports pytest.

Three layers are checked: the default function (T2), graphiti's own check in a
clean subprocess (T1), the call site inside ``_ensure_initialized`` (T3), and a
structural guard that no second ``Graphiti(...)`` construction can bypass it
(T4).
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

from docugraph.storage.graph_store import apply_telemetry_default

ROOT = Path(__file__).resolve().parents[2]
GRAPH_STORE = ROOT / "src" / "docugraph" / "storage" / "graph_store.py"
GRAPHITI_ENV = "GRAPHITI_TELEMETRY_ENABLED"

# No pytest import anywhere in this script: graphiti's own check consults
# sys.modules, and importing pytest would make the answer "off" for the wrong
# reason.
CHILD_SCRIPT = """
import os
import sys

from docugraph.storage.graph_store import apply_telemetry_default

apply_telemetry_default(os.environ)

from graphiti_core.telemetry.telemetry import is_telemetry_enabled

print("telemetry_enabled:", is_telemetry_enabled())
print("pytest_imported:", "pytest" in sys.modules)
"""


def _child_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key != GRAPHITI_ENV}
    return env


def _called_name(call: ast.Call) -> str | None:
    """The final name of a call target: `f(...)` and `mod.f(...)` both yield f."""
    target = call.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def test_t2_absent_variable_defaults_to_false() -> None:
    """ORACLE: T2 / §3.5 — absent becomes "false"."""
    environ: dict[str, str] = {}
    apply_telemetry_default(environ)
    assert environ[GRAPHITI_ENV] == "false"


def test_t2_explicit_value_is_left_alone() -> None:
    """ORACLE: T2 / §3.5 — an explicit "true" (or any value) is the user's call."""
    environ = {GRAPHITI_ENV: "true"}
    apply_telemetry_default(environ)
    assert environ[GRAPHITI_ENV] == "true"


def test_t1_graphiti_reports_telemetry_off_without_pytest() -> None:
    """ORACLE: T1 / §3.5 — graphiti's own check in a process that never ran pytest."""
    completed = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT],
        cwd=ROOT,
        env=_child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "telemetry_enabled: False" in completed.stdout, completed.stdout
    assert "pytest_imported: False" in completed.stdout, completed.stdout


def _ensure_initialized_node() -> ast.AST:
    tree = ast.parse(GRAPH_STORE.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_ensure_initialized"
        ):
            return node
    raise AssertionError(f"no _ensure_initialized in {GRAPH_STORE}")


def test_t3_the_telemetry_default_is_called_before_graphiti_is_constructed() -> None:
    """ORACLE: T3 / A-5 — removing the call must go red."""
    function = _ensure_initialized_node()
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]

    default_lines = [
        node.lineno for node in calls if _called_name(node) in {"apply_telemetry_default"}
    ]
    graphiti_lines = [node.lineno for node in calls if _called_name(node) == "Graphiti"]

    assert default_lines, (
        "_ensure_initialized does not call apply_telemetry_default; graphiti "
        "will send PostHog telemetry on every construction"
    )
    assert graphiti_lines, "_ensure_initialized no longer constructs Graphiti"
    assert min(default_lines) < min(graphiti_lines), (
        "apply_telemetry_default must run before the first Graphiti(...) call"
    )


def test_t4_graphiti_is_constructed_only_in_graph_store() -> None:
    """ORACLE: T4 / A-5 — a second construction site would bypass the default."""
    constructions: list[str] = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called_name(node) == "Graphiti":
                constructions.append(path.relative_to(ROOT).as_posix())

    assert constructions == ["src/docugraph/storage/graph_store.py"], (
        "Graphiti(...) must be constructed exactly once, in storage/graph_store.py; "
        f"found: {constructions}"
    )
