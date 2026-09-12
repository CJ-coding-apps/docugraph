"""Local file indexing for Markdown, RST, and text documentation.

Supports:
- Markdown (.md, .mdx)
- reStructuredText (.rst)
- Plain text (.txt)
- Code files with docstrings (optional)
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from docugraph.core.models import Document
from docugraph.ingestion.chunker import Chunker, ChunkerConfig


class FileType(StrEnum):
    """Supported file types."""

    MARKDOWN = "markdown"
    RST = "rst"
    TEXT = "text"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class LocalFileConfig:
    """Configuration for local file indexing."""

    # File patterns to include
    include_patterns: list[str] = field(default_factory=lambda: ["*.md", "*.mdx", "*.rst", "*.txt"])

    # File patterns to exclude
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "*.min.js",
            "*.min.css",
            "node_modules/*",
            ".git/*",
            "__pycache__/*",
            "*.pyc",
            ".venv/*",
            "venv/*",
            "dist/*",
            "build/*",
            ".tox/*",
            ".pytest_cache/*",
        ]
    )

    # Directories to skip entirely
    skip_dirs: list[str] = field(
        default_factory=lambda: [
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".tox",
            ".pytest_cache",
            "dist",
            "build",
            ".eggs",
        ]
    )

    # Maximum file size in bytes (default 1MB)
    max_file_size: int = 1024 * 1024

    # Whether to follow symlinks
    follow_symlinks: bool = False

    # Extract title from content
    extract_title: bool = True

    # Include hidden files (starting with .)
    include_hidden: bool = False

    # Encoding to use for reading files
    encoding: str = "utf-8"


@dataclass
class IndexedFile:
    """Represents an indexed file with metadata."""

    path: Path
    relative_path: str
    file_type: FileType
    content: str
    title: str | None
    size: int
    modified_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_document(self) -> Document:
        """Convert to Document model."""
        return Document(
            source_path=str(self.relative_path),
            title=self.title,
            content=self.content,
            content_type=self.file_type.value,
            metadata={
                "file_type": self.file_type.value,
                "size": self.size,
                "modified_at": self.modified_at.isoformat(),
                **self.metadata,
            },
        )


class LocalFileIndexer:
    """Indexes local files for documentation retrieval.

    Walks directory trees, reads supported file types,
    extracts content and metadata, and optionally chunks documents.
    """

    def __init__(
        self,
        config: LocalFileConfig | None = None,
        chunker: Chunker | None = None,
    ) -> None:
        """Initialize the local file indexer.

        Args:
            config: Indexer configuration
            chunker: Optional chunker for document splitting
        """
        self._config = config or LocalFileConfig()
        self._chunker = chunker

    def _detect_file_type(self, path: Path) -> FileType:
        """Detect file type from extension.

        Args:
            path: File path

        Returns:
            Detected file type
        """
        suffix = path.suffix.lower()

        if suffix in (".md", ".mdx", ".markdown"):
            return FileType.MARKDOWN
        elif suffix == ".rst":
            return FileType.RST
        elif suffix == ".txt":
            return FileType.TEXT
        elif suffix in (".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".h"):
            return FileType.CODE
        else:
            return FileType.UNKNOWN

    def _matches_pattern(self, path: Path, patterns: list[str]) -> bool:
        """Check if path matches any pattern.

        Args:
            path: File path
            patterns: Glob patterns

        Returns:
            True if matches any pattern
        """
        path_str = str(path)
        name = path.name

        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(path_str, pattern):
                return True
        return False

    def _should_skip_dir(self, dir_name: str) -> bool:
        """Check if directory should be skipped.

        Args:
            dir_name: Directory name

        Returns:
            True if should skip
        """
        if dir_name in self._config.skip_dirs:
            return True
        return bool(not self._config.include_hidden and dir_name.startswith("."))

    def _should_index_file(self, path: Path) -> bool:
        """Check if file should be indexed.

        Args:
            path: File path

        Returns:
            True if should index
        """
        # Check hidden
        if not self._config.include_hidden and path.name.startswith("."):
            return False

        # Check include patterns
        if not self._matches_pattern(path, self._config.include_patterns):
            return False

        # Check exclude patterns
        if self._matches_pattern(path, self._config.exclude_patterns):
            return False

        # Check size
        try:
            if path.stat().st_size > self._config.max_file_size:
                return False
        except OSError:
            return False

        return True

    def _extract_title_from_markdown(self, content: str) -> str | None:
        """Extract title from Markdown content.

        Looks for:
        1. YAML frontmatter title
        2. First H1 heading

        Args:
            content: Markdown content

        Returns:
            Extracted title or None
        """
        # Check for YAML frontmatter
        if content.startswith("---"):
            end = content.find("---", 3)
            if end > 0:
                frontmatter = content[3:end]
                title_match = re.search(
                    r"^title:\s*[\"']?(.+?)[\"']?\s*$", frontmatter, re.MULTILINE
                )
                if title_match:
                    return title_match.group(1).strip()

        # Look for first H1 heading
        h1_match = re.search(r"^#\s+(.+?)$", content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()

        return None

    def _extract_title_from_rst(self, content: str) -> str | None:
        """Extract title from RST content.

        Looks for the first title (underlined with = or -).

        Args:
            content: RST content

        Returns:
            Extracted title or None
        """
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if i > 0 and line and len(set(line)) == 1 and line[0] in "=-~^":
                # Previous line is the title
                title = lines[i - 1].strip()
                if title:
                    return title
        return None

    def _extract_title(self, content: str, file_type: FileType, path: Path) -> str | None:
        """Extract title from content based on file type.

        Args:
            content: File content
            file_type: File type
            path: File path

        Returns:
            Extracted title or filename-based title
        """
        if not self._config.extract_title:
            return path.stem

        title = None

        if file_type == FileType.MARKDOWN:
            title = self._extract_title_from_markdown(content)
        elif file_type == FileType.RST:
            title = self._extract_title_from_rst(content)

        # Fallback to filename
        if not title:
            title = path.stem.replace("-", " ").replace("_", " ").title()

        return title

    def _read_file(self, path: Path) -> str | None:
        """Read file content with encoding handling.

        Args:
            path: File path

        Returns:
            File content or None if unreadable
        """
        encodings = [self._config.encoding, "utf-8", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                return path.read_text(encoding=encoding)
            except (UnicodeDecodeError, LookupError):
                continue
            except OSError:
                return None

        return None

    def index_file(self, path: Path, base_path: Path | None = None) -> IndexedFile | None:
        """Index a single file.

        Args:
            path: File path to index
            base_path: Base path for computing relative paths

        Returns:
            IndexedFile or None if not indexable
        """
        path = path.resolve()

        if not path.is_file():
            return None

        if not self._should_index_file(path):
            return None

        content = self._read_file(path)
        if content is None:
            return None

        file_type = self._detect_file_type(path)
        title = self._extract_title(content, file_type, path)

        # Compute relative path
        if base_path:
            try:
                relative_path = str(path.relative_to(base_path))
            except ValueError:
                relative_path = str(path)
        else:
            relative_path = str(path)

        # Get file stats
        stat = path.stat()

        return IndexedFile(
            path=path,
            relative_path=relative_path,
            file_type=file_type,
            content=content,
            title=title,
            size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            metadata={
                "encoding": self._config.encoding,
            },
        )

    def index_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
    ) -> Iterator[IndexedFile]:
        """Index all supported files in a directory.

        Args:
            directory: Directory to index
            recursive: Whether to recurse into subdirectories

        Yields:
            IndexedFile for each indexed file
        """
        directory = Path(directory).resolve()

        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        if recursive:
            for root, dirs, files in os.walk(directory, followlinks=self._config.follow_symlinks):
                # Filter directories in place
                dirs[:] = [d for d in dirs if not self._should_skip_dir(d)]

                for filename in files:
                    path = Path(root) / filename
                    indexed = self.index_file(path, base_path=directory)
                    if indexed:
                        yield indexed
        else:
            for path in directory.iterdir():
                if path.is_file():
                    indexed = self.index_file(path, base_path=directory)
                    if indexed:
                        yield indexed

    def index_to_documents(
        self,
        directory: Path | str,
        recursive: bool = True,
    ) -> list[Document]:
        """Index directory and return Document objects.

        Args:
            directory: Directory to index
            recursive: Whether to recurse into subdirectories

        Returns:
            List of Document objects
        """
        documents = []
        for indexed_file in self.index_directory(directory, recursive):
            documents.append(indexed_file.to_document())
        return documents

    def index_and_chunk(
        self,
        directory: Path | str,
        recursive: bool = True,
    ) -> tuple[list[Document], list[Any]]:
        """Index directory and chunk documents.

        Args:
            directory: Directory to index
            recursive: Whether to recurse into subdirectories

        Returns:
            Tuple of (documents, chunks)
        """
        if self._chunker is None:
            self._chunker = Chunker(ChunkerConfig())

        documents = []
        all_chunks = []

        for indexed_file in self.index_directory(directory, recursive):
            doc = indexed_file.to_document()
            documents.append(doc)

            chunks = self._chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        return documents, all_chunks

    def get_stats(self, directory: Path | str, recursive: bool = True) -> dict[str, Any]:
        """Get statistics about indexable files in a directory.

        Args:
            directory: Directory to analyze
            recursive: Whether to recurse into subdirectories

        Returns:
            Statistics dict
        """
        stats: dict[str, Any] = {
            "total_files": 0,
            "total_size": 0,
            "by_type": {},
            "skipped": 0,
        }

        directory = Path(directory).resolve()

        for root, dirs, files in os.walk(directory, followlinks=self._config.follow_symlinks):
            dirs[:] = [d for d in dirs if not self._should_skip_dir(d)]

            for filename in files:
                path = Path(root) / filename

                if self._should_index_file(path):
                    stats["total_files"] += 1
                    try:
                        size = path.stat().st_size
                        stats["total_size"] += size
                    except OSError:
                        pass

                    file_type = self._detect_file_type(path).value
                    stats["by_type"][file_type] = stats["by_type"].get(file_type, 0) + 1
                else:
                    stats["skipped"] += 1

            if not recursive:
                break

        return stats


def get_local_file_indexer(
    config: LocalFileConfig | None = None,
    chunker: Chunker | None = None,
) -> LocalFileIndexer:
    """Factory function to get a LocalFileIndexer instance.

    Args:
        config: Optional configuration
        chunker: Optional chunker instance

    Returns:
        LocalFileIndexer instance
    """
    return LocalFileIndexer(config=config, chunker=chunker)
