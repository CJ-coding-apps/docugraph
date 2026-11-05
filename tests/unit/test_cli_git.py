"""G15: the CLI must surface a path-escape refusal as a non-zero exit.

A-10 puts ``src/docugraph/interfaces/cli.py`` in scope for exactly one change:
``RepoPathRefusedError`` and ``InvalidGitRefError`` become ``ClickException``
(exit 1). Every other CLI branch is unchanged. A refusal that escaped the root
was never a supported behaviour, so the CLI must not exit 0 there.
"""

import pytest
from click.testing import CliRunner

from docugraph.core.config import reset_config
from docugraph.ingestion.git_indexer import GitIndexer
from docugraph.interfaces.cli import main as cli


def test_g15_absolute_name_exits_nonzero_and_names_the_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORACLE: G15 / A-1 — non-zero exit, the path in the error, no git process."""
    outside = tmp_path / "outside-repo"
    outside.mkdir()
    monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", str(tmp_path / "data"))
    reset_config()

    git_calls: list[tuple] = []

    def _no_git(self, *args, **kwargs):
        git_calls.append(args)
        raise AssertionError("git must not run for a refused name")

    monkeypatch.setattr(GitIndexer, "_run_git_command", _no_git)

    result = CliRunner().invoke(
        cli,
        ["index", "git", "https://example.invalid/repo.git", "--name", str(outside)],
    )

    assert result.exit_code != 0, result.output
    assert str(outside) in result.output, result.output
    assert git_calls == [], "a path-escape refusal must precede every git process"
    assert not isinstance(result.exception, AssertionError), (
        "the escape reached git; the CLI must refuse before spawning it"
    )
