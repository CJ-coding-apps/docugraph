"""The fixture builder refuses both of the things it must refuse (WP §4.4).

``python -m docugraph.evals.build_graph_fixture`` regenerates a committed,
LLM-extracted fixture -- a destructive, expensive, non-reproducible act. The
work order makes it refuse unless it is given an explicit
``--i-am-regenerating`` flag *and* has a configured LLM. These tests prove both
refusals in child processes, so a partially imported module cannot leak state
into the test session.

No LLM key is set and no network call is made: the "no LLM" case is the
OpenAI provider with no key, which is refused locally by the preflight.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the builder in an isolated HOME and cwd.

    Isolation matters twice: a user config file must not supply an LLM, and a
    relative output path must not land in the repository. The provider is
    OpenAI with no key so the refusal is local and deterministic.
    """
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "PYTHONPATH": str(REPO / "src"),
            "DOCUGRAPH_STORAGE__DATA_DIR": str(tmp_path / "data"),
            "DOCUGRAPH_LLM__PROVIDER": "openai",
        }
    )
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, "-m", "docugraph.evals.build_graph_fixture", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=300,
    )


def test_refuses_without_the_regeneration_flag(tmp_path):
    proc = _run([], tmp_path)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        f"the builder exited 0 without --i-am-regenerating; output:\n{combined}"
    )
    assert "--i-am-regenerating" in combined, (
        f"the refusal does not name the explicit flag; output:\n{combined}"
    )


def _fixture_dir_state() -> list[tuple[str, int]]:
    fixture_dir = REPO / "benchmarks" / "retrieval" / "graph_fixture"
    if not fixture_dir.is_dir():
        return []
    return sorted((path.name, path.stat().st_mtime_ns) for path in fixture_dir.glob("*"))


def test_refuses_without_an_llm_configured(tmp_path):
    before = _fixture_dir_state()
    proc = _run(["--i-am-regenerating"], tmp_path)
    combined = (proc.stdout + proc.stderr).lower()
    assert proc.returncode != 0, f"the builder exited 0 with no LLM configured; output:\n{combined}"
    assert "llm" in combined, f"the refusal does not say an LLM is required; output:\n{combined}"
    assert "key" in combined or "configur" in combined, (
        f"the refusal does not say how to configure the LLM; output:\n{combined}"
    )
    assert _fixture_dir_state() == before, (
        "a refused run wrote into benchmarks/retrieval/graph_fixture"
    )


def test_refuses_a_dirty_src_tree(monkeypatch, tmp_path, capsys):
    """The manifest's extraction_git_sha must name the code that built it.

    The dirtiness is injected, so this runs in-process: a child process would
    read the developer's real tree and could pass or fail by accident.
    """
    from types import SimpleNamespace

    import docugraph.evals.build_graph_fixture as builder

    monkeypatch.setattr(builder, "get_llm_provider", lambda: SimpleNamespace())
    monkeypatch.setattr(
        builder,
        "_src_tree_dirty_reason",
        lambda: "src/ has uncommitted changes:\n M src/docugraph/core/llm.py",
    )

    def _must_not_run(_coro):
        raise AssertionError("the builder started regenerating from a dirty src/ tree")

    monkeypatch.setattr(builder.asyncio, "run", _must_not_run)
    output = tmp_path / "fixture"

    rc = builder.main(["--i-am-regenerating", "--output", str(output)])

    err = capsys.readouterr().err
    assert rc == 2
    assert "uncommitted changes" in err
    assert "src/" in err
    assert not output.exists()
