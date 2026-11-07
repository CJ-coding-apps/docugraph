"""Git indexer guards: ref validation, argv assembly, and A-1 path escape.

The pure rows (G8–G10, G13, G16) call the builders directly. The handler rows
(G11/G12) go through ``_index_git`` with a fake resolver returning a global
address, so no test here touches the network. Git processes are forbidden by a
patched ``_run_git_command`` that raises ``AssertionError``; a refusal that
somehow fell through to git would fail the test rather than clone anything.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from docugraph.core.config import reset_config
from docugraph.ingestion.git_indexer import (
    GitIndexer,
    GitIndexerConfig,
    RepoPathRefusedError,
    build_checkout_args,
    build_clone_args,
    is_valid_git_ref,
    validate_repo_name,
)
from docugraph.interfaces import mcp_server
from docugraph.security.url_policy import RefusalReason, UrlKind, UrlPolicy

URL = "https://example.invalid/repo.git"
GIT = shutil.which("git")
requires_git = pytest.mark.skipif(GIT is None, reason="git executable not found on PATH")

# The three config flags A-4 adds under policy, in order.
RESTRICTED_CONFIG_PAIRS = [
    ["-c", "protocol.file.allow=never"],
    ["-c", "protocol.ext.allow=never"],
    ["-c", "http.followRedirects=false"],
]


def _git(*args: str) -> None:
    subprocess.run(
        [GIT, *args],  # type: ignore[list-item]
        check=True,
        capture_output=True,
        text=True,
    )


def _result_text(result) -> str:
    return "\n".join(
        item.text for item in getattr(result, "content", []) if getattr(item, "text", None)
    )


def _occurs_before(args: list[str], sequence: list[str], stop: int) -> bool:
    width = len(sequence)
    return any(args[i : i + width] == sequence for i in range(stop - width + 1))


def _no_git(self, *args, **kwargs):
    raise AssertionError("git must not run before the refusal")


def _no_store():
    raise AssertionError("the vector store must not be touched by a refusal")


@pytest.fixture
def fake_git_policy() -> UrlPolicy:
    """A git-remote policy whose resolver answers every host with a global IP."""
    return UrlPolicy(kind=UrlKind.GIT_REMOTE, resolver=lambda host: ("1.1.1.1",))


@pytest.fixture
def outside_repo(tmp_path: Path) -> Path:
    """An existing repo outside repos_dir, with a local bare origin (A-10).

    The origin makes 0.2.0's ``git pull --ff-only`` possible offline, so the red
    run measures the escape rather than a network error.
    """
    bare = tmp_path / "origin.git"
    repo = tmp_path / "outside-repo"
    _git("init", "--bare", str(bare))
    _git("init", str(repo))
    _git("-C", str(repo), "remote", "add", "origin", str(bare))
    return repo


# ---------------------------------------------------------------------------
# G8/G9: ref validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ref", ["-x", "--orphan=y", "a..b", "bad ref"])
def test_g8_invalid_refs_are_rejected(ref: str) -> None:
    """ORACLE: G8 / git-check-ref-format(1)."""
    assert not is_valid_git_ref(ref)


@pytest.mark.parametrize("ref", ["main", "v0.2.0", "feature/x", "0123abcd"])
def test_g9_valid_refs_are_accepted(ref: str) -> None:
    """ORACLE: G9 / git-check-ref-format(1)."""
    assert is_valid_git_ref(ref)


def test_checkout_args_read_the_ref_as_a_revision_not_a_path() -> None:
    """ORACLE: G8/G9 — `--` after the ref keeps git from taking it as a path."""
    assert build_checkout_args("main") == ["checkout", "main", "--"]


# ---------------------------------------------------------------------------
# G10/G13/G16: clone argv (pure)
# ---------------------------------------------------------------------------


def test_g10_clone_argv_places_double_dash_immediately_before_url() -> None:
    """ORACLE: G10 / git-clone(1) — `--` ends options before <url> <path>."""
    local_path = Path("/data/repos/ok-name")
    args = build_clone_args(
        url=URL,
        local_path=local_path,
        depth=None,
        ref=None,
        restrict_protocols=False,
    )
    assert args[-3:] == ["--", URL, str(local_path)]
    assert "clone" in args


def test_g13_accepted_name_builds_the_path_under_repos_dir() -> None:
    """ORACLE: G13 / A-1 — an ordinary name stays inside the configured root."""
    repos_dir = Path("/data/repos")
    args = build_clone_args(
        url=URL,
        local_path=repos_dir / "ok-name",
        depth=1,
        ref="main",
        restrict_protocols=False,
    )
    assert args[-1] == str(repos_dir / "ok-name")
    assert args[-3:] == ["--", URL, str(repos_dir / "ok-name")]


def test_g16_policy_adds_redirect_lockdown_before_clone() -> None:
    """ORACLE: G16 / A-4 — the -c flags precede the `clone` subcommand."""
    args = build_clone_args(
        url=URL,
        local_path=Path("/data/repos/ok-name"),
        depth=1,
        ref="main",
        restrict_protocols=True,
    )
    clone_at = args.index("clone")
    for pair in RESTRICTED_CONFIG_PAIRS:
        assert _occurs_before(args, pair, clone_at), f"missing {pair!r} before clone in {args!r}"
    assert args[-3:] == ["--", URL, "/data/repos/ok-name"]


# ---------------------------------------------------------------------------
# validate_repo_name (A-1, pure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b", "a\x00b", "/abs"])
def test_a1_validate_repo_name_rejects_escaping_components(name: str) -> None:
    """ORACLE: A-1 / N3 — one non-empty path component, no separators, no NUL."""
    refusal = validate_repo_name(name)
    assert refusal is not None
    assert refusal.reason is RefusalReason.PATH_ESCAPE


def test_a1_validate_repo_name_accepts_an_ordinary_name() -> None:
    """ORACLE: A-1 / G13."""
    assert validate_repo_name("ok-name") is None


# ---------------------------------------------------------------------------
# G11/G12: handler-level refusals, no git and no store
# ---------------------------------------------------------------------------


@requires_git
async def test_g11_absolute_name_is_refused_before_git(
    tmp_path: Path,
    outside_repo: Path,
    fake_git_policy: UrlPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ORACLE: G11 / A-1 — an absolute name must not reach git or the store."""
    monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", str(tmp_path / "data"))
    reset_config()
    monkeypatch.setattr(mcp_server, "_git_policy", lambda: fake_git_policy)
    monkeypatch.setattr(mcp_server, "get_vector_store", _no_store)
    monkeypatch.setattr(GitIndexer, "_run_git_command", _no_git)

    result = await mcp_server._index_git({"url": URL, "name": str(outside_repo)})

    text = _result_text(result)
    assert result.isError
    assert text.startswith("Refused: path_escape"), text


async def test_g12_relative_dotdot_name_is_refused_before_git(
    tmp_path: Path,
    fake_git_policy: UrlPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ORACLE: G12 / A-1 — `..` is not a name, whether or not the target exists."""
    monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", str(tmp_path / "data"))
    reset_config()
    monkeypatch.setattr(mcp_server, "_git_policy", lambda: fake_git_policy)
    monkeypatch.setattr(mcp_server, "get_vector_store", _no_store)
    monkeypatch.setattr(GitIndexer, "_run_git_command", _no_git)

    result = await mcp_server._index_git({"url": URL, "name": "../outside-repo"})

    text = _result_text(result)
    assert result.isError
    assert text.startswith("Refused: path_escape"), text


# ---------------------------------------------------------------------------
# G14 and the derived name: escapes caught by the shell / by the rule
# ---------------------------------------------------------------------------


@requires_git
def test_g14_symlink_inside_repos_dir_is_refused(
    tmp_path: Path,
    outside_repo: Path,
    fake_git_policy: UrlPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ORACLE: G14 / A-1 — resolve() catches a symlink planted in repos_dir."""
    repos_dir = tmp_path / "repos"
    repos_dir.mkdir()
    link = repos_dir / "link"
    try:
        link.symlink_to(outside_repo, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platform-dependent
        pytest.skip(f"filesystem cannot create a symlink: {exc}")

    monkeypatch.setattr(GitIndexer, "_run_git_command", _no_git)
    indexer = GitIndexer(config=GitIndexerConfig(repos_dir=repos_dir), url_policy=fake_git_policy)

    with pytest.raises(RepoPathRefusedError):
        indexer.clone(URL, name="link")


def test_derived_dotdot_name_is_refused_not_renamed(
    tmp_path: Path,
    fake_git_policy: UrlPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ORACLE: A-1 / N4 — a URL deriving `..` is refused; it is not repaired."""
    repos_dir = tmp_path / "repos"
    repos_dir.mkdir()
    monkeypatch.setattr(GitIndexer, "_run_git_command", _no_git)
    indexer = GitIndexer(config=GitIndexerConfig(repos_dir=repos_dir), url_policy=fake_git_policy)

    with pytest.raises(RepoPathRefusedError):
        indexer.clone("https://host/a/..")


@requires_git
def test_a1_refusal_precedes_update_on_an_existing_path(
    tmp_path: Path,
    outside_repo: Path,
    fake_git_policy: UrlPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ORACLE: A-1 / N10 — an existing escaping path must not reach update()."""

    def _update_must_not_run(self, *args, **kwargs):
        raise AssertionError("update() ran on an escaping path")

    repos_dir = tmp_path / "repos"
    repos_dir.mkdir()
    monkeypatch.setattr(GitIndexer, "update", _update_must_not_run)
    monkeypatch.setattr(GitIndexer, "_run_git_command", _no_git)
    indexer = GitIndexer(config=GitIndexerConfig(repos_dir=repos_dir), url_policy=fake_git_policy)

    with pytest.raises(RepoPathRefusedError):
        indexer.clone(URL, name=str(outside_repo))


# ---------------------------------------------------------------------------
# Round-1 review: the update path (MED-1 / G17, G18) and `a/b` (LOW-1)
# ---------------------------------------------------------------------------


def test_a1_multi_component_name_is_refused_before_git(
    tmp_path: Path,
    fake_git_policy: UrlPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ORACLE: LOW-1 / A-1 — `a/b` is two components; resolve() alone admits it.

    ``repos_dir/a/b`` resolves *inside* the repos directory when ``b`` does not
    exist, so the containment stage cannot catch this name. Only the
    single-component rule can, and it must run before git.
    """
    repos_dir = tmp_path / "repos"
    repos_dir.mkdir()
    monkeypatch.setattr(GitIndexer, "_run_git_command", _no_git)
    indexer = GitIndexer(config=GitIndexerConfig(repos_dir=repos_dir), url_policy=fake_git_policy)

    with pytest.raises(RepoPathRefusedError) as excinfo:
        indexer.clone(URL, name="a/b")

    assert "Refused: path_escape" in str(excinfo.value)


@requires_git
def test_g17_existing_repo_with_a_different_origin_is_refused(
    tmp_path: Path,
    outside_repo: Path,
    fake_git_policy: UrlPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ORACLE: MED-1 / G17 — pull would use the origin, so the origin is what is judged.

    The requested URL passes the policy; the existing clone's origin is a
    different value. Refuse before any pull or fetch.
    """
    repos_dir = tmp_path / "repos"
    repos_dir.mkdir()
    existing = repos_dir / "existing"
    _git("init", str(existing))
    _git("-C", str(existing), "remote", "add", "origin", str(outside_repo))

    calls: list[list[str]] = []
    real = GitIndexer._run_git_command

    def _record(self, args, **kwargs):
        calls.append(list(args))
        return real(self, args, **kwargs)

    monkeypatch.setattr(GitIndexer, "_run_git_command", _record)
    indexer = GitIndexer(config=GitIndexerConfig(repos_dir=repos_dir), url_policy=fake_git_policy)

    with pytest.raises(RepoPathRefusedError) as excinfo:
        indexer.clone(URL, name="existing")

    assert "origin differs" in str(excinfo.value), str(excinfo.value)
    assert not any("pull" in args or "fetch" in args for args in calls), (
        f"a refused origin must not be pulled: {calls!r}"
    )


def test_g18_pull_carries_the_restricted_config_under_policy(
    tmp_path: Path,
    fake_git_policy: UrlPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ORACLE: MED-1 / G18 — pull and the fetch fallback carry the same -c lockdown."""
    repos_dir = tmp_path / "repos"
    existing = repos_dir / "existing"
    (existing / ".git").mkdir(parents=True)

    calls: list[list[str]] = []

    def _fake(self, args, cwd=None, timeout=None):
        calls.append(list(args))
        if args[:3] == ["remote", "get-url", "origin"]:
            return URL, "", 0
        return "", "", 0

    monkeypatch.setattr(GitIndexer, "_run_git_command", _fake)
    indexer = GitIndexer(config=GitIndexerConfig(repos_dir=repos_dir), url_policy=fake_git_policy)

    indexer.update(existing, expected_url=URL)

    pulls = [args for args in calls if "pull" in args]
    assert pulls, f"update() never ran pull: {calls!r}"
    pull = pulls[0]
    pull_at = pull.index("pull")
    for pair in RESTRICTED_CONFIG_PAIRS:
        assert _occurs_before(pull, pair, pull_at), f"missing {pair!r} before pull in {pull!r}"


# ---------------------------------------------------------------------------
# A-2 (0.3.0): a remote that resolves to metadata is refused under allow_private
# ---------------------------------------------------------------------------


async def test_p14_a_git_remote_resolving_to_metadata_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ORACLE: A-2 / P14 — allow_private never re-enables a metadata address."""
    monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", str(tmp_path / "data"))
    reset_config()
    policy = UrlPolicy(
        kind=UrlKind.GIT_REMOTE,
        allow_private=True,
        resolver=lambda host: ("169.254.169.254",),
    )
    monkeypatch.setattr(mcp_server, "_git_policy", lambda: policy)
    monkeypatch.setattr(mcp_server, "get_vector_store", _no_store)
    monkeypatch.setattr(GitIndexer, "_run_git_command", _no_git)

    result = await mcp_server._index_git(
        {"url": "https://example.invalid/repo.git", "name": "repo"}
    )

    text = _result_text(result)
    assert result.isError
    assert text.startswith("Refused: private_address"), text
