"""Git repository indexing for documentation and code.

Supports:
- Cloning repositories (HTTPS, SSH)
- Indexing documentation files
- Optional code file indexing with docstring extraction
- Branch/tag checkout
- Incremental updates via git pull
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 -- git CLI wrapper; argv is a list, never shell
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docugraph.core.config import get_config
from docugraph.core.models import Document
from docugraph.ingestion.chunker import Chunker, ChunkerConfig
from docugraph.ingestion.local_files import (
    IndexedFile,
    LocalFileConfig,
    LocalFileIndexer,
)
from docugraph.security.url_policy import (
    RefusalReason,
    UrlPolicy,
    UrlRefusal,
    UrlRefusedError,
    format_refusal,
)

# Always applied under a URL policy (A-4): local and ext transports have no
# place in a remote clone, and a redirect would hide the URL that was judged.
_RESTRICTED_CONFIG_PAIRS = (
    ("-c", "protocol.file.allow=never"),
    ("-c", "protocol.ext.allow=never"),
    ("-c", "http.followRedirects=false"),
)


def is_valid_git_ref(ref: str) -> bool:
    """Whether ``ref`` is a legal revision name (git-check-ref-format shaped).

    A value that begins with ``-`` would be read as an option, so it is
    rejected here as well as by git.
    """
    if not ref or ref.startswith("-") or ref == "@":
        return False
    if ref.endswith("/") or ref.endswith(".") or "//" in ref or ".." in ref:
        return False
    if "@{" in ref or "\\" in ref:
        return False
    for char in ref:
        if ord(char) <= 0x20 or ord(char) == 0x7F or char in "~^:?*[":
            return False
    for component in ref.split("/"):
        if not component or component.startswith(".") or component.endswith(".lock"):
            return False
    return True


def validate_repo_name(name: str) -> UrlRefusal | None:
    """One non-empty path component that cannot leave the repos directory.

    A derived name is judged by the same rule; it is refused, never repaired.
    """
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or Path(name).is_absolute()
    ):
        return UrlRefusal(
            RefusalReason.PATH_ESCAPE,
            f"repository name {name!r} escapes the repos directory",
        )
    return None


def _restricted_config(restrict_protocols: bool) -> list[str]:
    """The ``-c`` pairs, in order, when the policy is in force."""
    args: list[str] = []
    if restrict_protocols:
        for pair in _RESTRICTED_CONFIG_PAIRS:
            args.extend(pair)
    return args


def build_clone_args(
    *,
    url: str,
    local_path: Path,
    depth: int | None,
    ref: str | None,
    restrict_protocols: bool,
) -> list[str]:
    """Assemble ``git clone`` argv; ``--`` ends options before url and path."""
    args = _restricted_config(restrict_protocols)

    args.append("clone")
    if depth is not None:
        args.extend(["--depth", str(depth)])
    if ref and depth:
        # For shallow clones the branch has to be named at clone time.
        args.extend(["--branch", ref])
    args.extend(["--", url, str(local_path)])
    return args


def build_checkout_args(ref: str) -> list[str]:
    """``git checkout <ref> --`` so git reads a revision, never a path."""
    return ["checkout", ref, "--"]


def build_pull_args(*, restrict_protocols: bool) -> list[str]:
    """``git pull --ff-only`` under the same lockdown as clone (MED-1)."""
    return [*_restricted_config(restrict_protocols), "pull", "--ff-only"]


def build_fetch_args(*, restrict_protocols: bool) -> list[str]:
    """The pull fallback, locked down the same way."""
    return [*_restricted_config(restrict_protocols), "fetch", "origin"]


class InvalidGitRefError(RuntimeError):
    """The requested ref is not a legal git revision name."""


class RepoPathRefusedError(RuntimeError):
    """The clone destination would leave the repos directory."""

    def __init__(self, refusal: UrlRefusal) -> None:
        self.refusal = refusal
        super().__init__(format_refusal(refusal))


@dataclass
class GitIndexerConfig:
    """Configuration for git repository indexing."""

    # Clone settings
    clone_depth: int | None = None  # None for full clone, 1 for shallow
    clone_timeout: int = 300  # seconds

    # Branch/ref to checkout
    ref: str | None = None  # branch, tag, or commit hash

    # Where to store cloned repos
    repos_dir: Path | None = None  # Default: config.storage.data_dir / "repos"

    # File indexing settings
    include_docs: bool = True  # Index documentation files
    include_code: bool = False  # Index code files with docstrings

    # Documentation patterns (in addition to LocalFileConfig defaults)
    doc_patterns: list[str] = field(
        default_factory=lambda: [
            "*.md",
            "*.mdx",
            "*.rst",
            "*.txt",
            "README*",
            "CHANGELOG*",
            "CONTRIBUTING*",
            "docs/**/*",
            "documentation/**/*",
        ]
    )

    # Code patterns (when include_code=True)
    code_patterns: list[str] = field(
        default_factory=lambda: [
            "*.py",
            "*.js",
            "*.ts",
            "*.java",
            "*.go",
            "*.rs",
        ]
    )

    # Directories to skip
    skip_dirs: list[str] = field(
        default_factory=lambda: [
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            "dist",
            "build",
            ".tox",
            "target",
            "vendor",
        ]
    )

    # Maximum file size (1MB default)
    max_file_size: int = 1024 * 1024

    # Clean up cloned repos after indexing
    cleanup_after_index: bool = False


@dataclass
class RepoInfo:
    """Information about a git repository."""

    url: str
    name: str
    local_path: Path
    branch: str | None
    commit_hash: str | None
    clone_time: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class GitIndexer:
    """Indexes git repositories for documentation and code.

    Clones repositories, indexes documentation files, and optionally
    extracts code documentation (docstrings, comments).
    """

    def __init__(
        self,
        config: GitIndexerConfig | None = None,
        chunker: Chunker | None = None,
        url_policy: UrlPolicy | None = None,
    ) -> None:
        """Initialize the git indexer.

        Args:
            config: Indexer configuration
            chunker: Optional chunker for document splitting
            url_policy: Optional remote URL policy. Its presence also turns on
                the protocol and redirect restrictions in the git argv.
        """
        self._config = config or GitIndexerConfig()
        self._chunker = chunker
        self._url_policy = url_policy

        # Set up repos directory
        if self._config.repos_dir is None:
            app_config = get_config()
            self._repos_dir = app_config.storage.data_dir / "repos"
        else:
            self._repos_dir = self._config.repos_dir

        self._repos_dir.mkdir(parents=True, exist_ok=True)

    def _extract_repo_name(self, url: str) -> str:
        """Extract repository name from URL.

        Args:
            url: Git repository URL

        Returns:
            Repository name
        """
        # Handle various URL formats
        # https://github.com/user/repo.git
        # git@github.com:user/repo.git
        # https://github.com/user/repo

        name = url.rstrip("/")
        if name.endswith(".git"):
            name = name[:-4]

        # Extract last path component
        if "/" in name:
            name = name.split("/")[-1]
        elif ":" in name:
            name = name.split(":")[-1].split("/")[-1]

        return name or "repo"

    def _run_git_command(
        self,
        args: list[str],
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> tuple[str, str, int]:
        """Run a git command.

        Args:
            args: Git command arguments
            cwd: Working directory
            timeout: Command timeout in seconds

        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        cmd = ["git"] + args
        timeout = timeout or self._config.clone_timeout

        try:
            result = subprocess.run(  # nosec B603 -- argv list, no shell; git only
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out", 1
        except Exception as e:
            return "", str(e), 1

    def _get_current_commit(self, repo_path: Path) -> str | None:
        """Get the current commit hash.

        Args:
            repo_path: Repository path

        Returns:
            Commit hash or None
        """
        stdout, _, code = self._run_git_command(
            ["rev-parse", "HEAD"],
            cwd=repo_path,
            timeout=10,
        )
        if code == 0:
            return stdout.strip()
        return None

    def _get_current_branch(self, repo_path: Path) -> str | None:
        """Get the current branch name.

        Args:
            repo_path: Repository path

        Returns:
            Branch name or None (if detached HEAD)
        """
        stdout, _, code = self._run_git_command(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            timeout=10,
        )
        if code == 0:
            branch = stdout.strip()
            return None if branch == "HEAD" else branch
        return None

    def clone(
        self,
        url: str,
        name: str | None = None,
        ref: str | None = None,
    ) -> RepoInfo:
        """Clone a git repository.

        Args:
            url: Repository URL
            name: Optional custom name (defaults to extracted from URL)
            ref: Branch, tag, or commit to checkout

        Returns:
            RepoInfo with repository details

        Raises:
            RuntimeError: If clone fails
        """
        if ref is not None and not is_valid_git_ref(ref):
            raise InvalidGitRefError(f"Invalid git ref: {ref!r}")

        name = name or self._extract_repo_name(url)

        # A-1: the name and the resolved destination are judged before any git
        # process, so an escaping path can never reach clone or update.
        refusal = validate_repo_name(name)
        if refusal is not None:
            raise RepoPathRefusedError(refusal)

        local_path = self._repos_dir / name
        repos_root = self._repos_dir.resolve()
        if not local_path.resolve().is_relative_to(repos_root):
            raise RepoPathRefusedError(
                UrlRefusal(
                    RefusalReason.PATH_ESCAPE,
                    f"{local_path} resolves outside {repos_root}",
                )
            )

        if self._url_policy is not None:
            url_refusal = self._url_policy.check(url)
            if url_refusal is not None:
                raise UrlRefusedError(url_refusal)

        # Check if already cloned
        if local_path.exists():
            # Update existing repo. The URL just judged is not necessarily the
            # one pull will use: update() reads the existing origin (MED-1).
            return self.update(local_path, expected_url=url)

        clone_args = build_clone_args(
            url=url,
            local_path=local_path,
            depth=self._config.clone_depth,
            ref=ref,
            restrict_protocols=self._url_policy is not None,
        )

        # Execute clone
        stdout, stderr, code = self._run_git_command(clone_args)

        if code != 0:
            raise RuntimeError(f"Failed to clone {url}: {stderr}")

        # Checkout specific ref if needed (and not already done via --branch)
        if ref and not self._config.clone_depth:
            checkout_args = build_checkout_args(ref)
            _, stderr, code = self._run_git_command(checkout_args, cwd=local_path)
            if code != 0:
                raise RuntimeError(f"Failed to checkout {ref}: {stderr}")

        return RepoInfo(
            url=url,
            name=name,
            local_path=local_path,
            branch=self._get_current_branch(local_path),
            commit_hash=self._get_current_commit(local_path),
            clone_time=datetime.now(UTC),
            metadata={"url": url, "ref": ref},
        )

    def update(
        self,
        repo_path: Path | str,
        *,
        expected_url: str | None = None,
    ) -> RepoInfo:
        """Update an existing repository with git pull.

        Args:
            repo_path: Path to repository
            expected_url: The URL the caller judged. Under a policy, the
                existing origin must equal it; pull would otherwise run against
                a remote that was never checked (MED-1).

        Returns:
            Updated RepoInfo

        Raises:
            RuntimeError: If update fails
        """
        repo_path = Path(repo_path).resolve()

        if not (repo_path / ".git").exists():
            raise RuntimeError(f"Not a git repository: {repo_path}")

        # Get remote URL
        stdout, _, code = self._run_git_command(
            ["remote", "get-url", "origin"],
            cwd=repo_path,
            timeout=10,
        )
        url = stdout.strip() if code == 0 else "unknown"

        policy = self._url_policy
        restrict_protocols = policy is not None
        if policy is not None:
            # pull uses the origin, not the requested URL, so the origin is
            # what has to match and pass the policy before any request is made.
            if expected_url is not None and url != expected_url:
                raise RepoPathRefusedError(
                    UrlRefusal(
                        RefusalReason.PATH_ESCAPE,
                        "existing repository's origin differs",
                    )
                )
            origin_refusal = policy.check(url)
            if origin_refusal is not None:
                raise UrlRefusedError(origin_refusal)

        # Pull updates
        stdout, stderr, code = self._run_git_command(
            build_pull_args(restrict_protocols=restrict_protocols),
            cwd=repo_path,
        )

        if code != 0:
            # Try fetch + reset as fallback
            self._run_git_command(
                build_fetch_args(restrict_protocols=restrict_protocols),
                cwd=repo_path,
            )

        return RepoInfo(
            url=url,
            name=repo_path.name,
            local_path=repo_path,
            branch=self._get_current_branch(repo_path),
            commit_hash=self._get_current_commit(repo_path),
            clone_time=datetime.now(UTC),
            metadata={"updated": True},
        )

    def _build_local_file_config(self) -> LocalFileConfig:
        """Build LocalFileConfig for indexing.

        Returns:
            LocalFileConfig with appropriate patterns
        """
        patterns = []

        if self._config.include_docs:
            patterns.extend(self._config.doc_patterns)

        if self._config.include_code:
            patterns.extend(self._config.code_patterns)

        return LocalFileConfig(
            include_patterns=patterns,
            exclude_patterns=[
                "*.min.js",
                "*.min.css",
                "*.map",
                "*.lock",
                "package-lock.json",
                "yarn.lock",
                "Cargo.lock",
            ],
            skip_dirs=self._config.skip_dirs,
            max_file_size=self._config.max_file_size,
            follow_symlinks=False,
            extract_title=True,
            include_hidden=False,
        )

    def index_repo(
        self,
        repo_path: Path | str,
        repo_info: RepoInfo | None = None,
    ) -> Iterator[IndexedFile]:
        """Index files in a repository.

        Args:
            repo_path: Path to repository
            repo_info: Optional repository info for metadata

        Yields:
            IndexedFile for each indexed file
        """
        repo_path = Path(repo_path).resolve()

        local_config = self._build_local_file_config()
        indexer = LocalFileIndexer(config=local_config)

        for indexed_file in indexer.index_directory(repo_path, recursive=True):
            # Add repository metadata
            if repo_info:
                indexed_file.metadata.update(
                    {
                        "repo_url": repo_info.url,
                        "repo_name": repo_info.name,
                        "branch": repo_info.branch,
                        "commit": repo_info.commit_hash,
                    }
                )

            yield indexed_file

    def clone_and_index(
        self,
        url: str,
        name: str | None = None,
        ref: str | None = None,
    ) -> tuple[RepoInfo, list[Document]]:
        """Clone a repository and index its contents.

        Args:
            url: Repository URL
            name: Optional custom name
            ref: Branch, tag, or commit to checkout

        Returns:
            Tuple of (RepoInfo, list of Documents)
        """
        # Clone repository
        repo_info = self.clone(url, name=name, ref=ref or self._config.ref)

        # Index files
        documents = []
        for indexed_file in self.index_repo(repo_info.local_path, repo_info):
            documents.append(indexed_file.to_document())

        # Cleanup if configured
        if self._config.cleanup_after_index:
            self.cleanup(repo_info.local_path)

        return repo_info, documents

    def clone_and_chunk(
        self,
        url: str,
        name: str | None = None,
        ref: str | None = None,
    ) -> tuple[RepoInfo, list[Document], list[Any]]:
        """Clone a repository, index, and chunk its contents.

        Args:
            url: Repository URL
            name: Optional custom name
            ref: Branch, tag, or commit to checkout

        Returns:
            Tuple of (RepoInfo, documents, chunks)
        """
        if self._chunker is None:
            self._chunker = Chunker(ChunkerConfig())

        repo_info, documents = self.clone_and_index(url, name=name, ref=ref)

        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self._chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        return repo_info, documents, all_chunks

    def cleanup(self, repo_path: Path | str) -> bool:
        """Remove a cloned repository.

        Args:
            repo_path: Path to repository

        Returns:
            True if cleanup succeeded
        """
        repo_path = Path(repo_path).resolve()

        if not repo_path.exists():
            return True

        # Safety check - must be under repos_dir
        try:
            repo_path.relative_to(self._repos_dir)
        except ValueError:
            return False

        try:
            shutil.rmtree(repo_path)
            return True
        except Exception:
            return False

    def list_repos(self) -> list[RepoInfo]:
        """List all cloned repositories.

        Returns:
            List of RepoInfo for each cloned repository
        """
        repos = []

        for path in self._repos_dir.iterdir():
            if path.is_dir() and (path / ".git").exists():
                # Get remote URL
                stdout, _, code = self._run_git_command(
                    ["remote", "get-url", "origin"],
                    cwd=path,
                    timeout=10,
                )
                url = stdout.strip() if code == 0 else "unknown"

                repos.append(
                    RepoInfo(
                        url=url,
                        name=path.name,
                        local_path=path,
                        branch=self._get_current_branch(path),
                        commit_hash=self._get_current_commit(path),
                        clone_time=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
                    )
                )

        return repos

    def get_repo_stats(self, repo_path: Path | str) -> dict[str, Any]:
        """Get statistics about a repository.

        Args:
            repo_path: Path to repository

        Returns:
            Statistics dict
        """
        repo_path = Path(repo_path).resolve()

        local_config = self._build_local_file_config()
        indexer = LocalFileIndexer(config=local_config)

        return indexer.get_stats(repo_path, recursive=True)


def get_git_indexer(
    config: GitIndexerConfig | None = None,
    chunker: Chunker | None = None,
    url_policy: UrlPolicy | None = None,
) -> GitIndexer:
    """Factory function to get a GitIndexer instance.

    Args:
        config: Optional configuration
        chunker: Optional chunker instance
        url_policy: Optional remote URL policy

    Returns:
        GitIndexer instance
    """
    return GitIndexer(config=config, chunker=chunker, url_policy=url_policy)
