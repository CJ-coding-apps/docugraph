"""Memory store for agent session state with repository/branch awareness."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docugraph.core.config import get_config
from docugraph.core.models import MemoryEntry


class MemoryStore:
    """Agent memory store backed by SQLite.

    Provides persistent key-value storage with:
    - Repository and branch awareness
    - Access time tracking (for LRU eviction)
    - Arbitrary value and context storage
    - Pattern-based search
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize the memory store.

        Args:
            db_path: Path to SQLite database. If None, uses config default.
        """
        if db_path is None:
            config = get_config()
            db_path = config.storage.data_dir / "memory.db"

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    repository TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    context TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    UNIQUE(repository, branch, key)
                )
            """)
            # Create indices for common query patterns
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_repo_branch
                ON memories(repository, branch)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_accessed
                ON memories(accessed_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_key_pattern
                ON memories(repository, branch, key)
            """)
            conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def set(
        self,
        key: str,
        value: Any,
        repository: str = "default",
        branch: str = "main",
        context: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a value in memory.

        Args:
            key: Memory key
            value: Value to store (will be JSON serialized)
            repository: Repository scope
            branch: Branch scope
            context: Optional context metadata

        Returns:
            The created/updated MemoryEntry
        """
        now = datetime.now(UTC)
        context = context or {}

        # Serialize value and context
        value_json = json.dumps(value)
        context_json = json.dumps(context)

        with self._get_connection() as conn:
            # Try to get existing entry
            cursor = conn.execute(
                "SELECT id FROM memories WHERE repository = ? AND branch = ? AND key = ?",
                (repository, branch, key),
            )
            row = cursor.fetchone()

            if row:
                # Update existing
                entry_id = row["id"]
                conn.execute(
                    """
                    UPDATE memories
                    SET value = ?, context = ?, accessed_at = ?
                    WHERE id = ?
                    """,
                    (value_json, context_json, now.isoformat(), entry_id),
                )
            else:
                # Insert new
                from docugraph.core.models import generate_id

                entry_id = generate_id()
                conn.execute(
                    """
                    INSERT INTO memories (id, repository, branch, key, value, context, created_at, accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry_id,
                        repository,
                        branch,
                        key,
                        value_json,
                        context_json,
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )

            conn.commit()

        return MemoryEntry(
            id=entry_id,
            repository=repository,
            branch=branch,
            key=key,
            value=value,
            context=context,
            created_at=now,
            accessed_at=now,
        )

    def get(
        self,
        key: str,
        repository: str = "default",
        branch: str = "main",
        default: Any = None,
    ) -> Any:
        """Retrieve a value from memory.

        Updates the accessed_at timestamp for LRU tracking.

        Args:
            key: Memory key
            repository: Repository scope
            branch: Branch scope
            default: Default value if not found

        Returns:
            The stored value or default
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, value FROM memories WHERE repository = ? AND branch = ? AND key = ?",
                (repository, branch, key),
            )
            row = cursor.fetchone()

            if row is None:
                return default

            # Update accessed_at
            now = datetime.now(UTC)
            conn.execute(
                "UPDATE memories SET accessed_at = ? WHERE id = ?",
                (now.isoformat(), row["id"]),
            )
            conn.commit()

            return json.loads(row["value"])

    def get_entry(
        self,
        key: str,
        repository: str = "default",
        branch: str = "main",
    ) -> MemoryEntry | None:
        """Retrieve a full MemoryEntry from memory.

        Args:
            key: Memory key
            repository: Repository scope
            branch: Branch scope

        Returns:
            MemoryEntry or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE repository = ? AND branch = ? AND key = ?",
                (repository, branch, key),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Update accessed_at
            now = datetime.now(UTC)
            conn.execute(
                "UPDATE memories SET accessed_at = ? WHERE id = ?",
                (now.isoformat(), row["id"]),
            )
            conn.commit()

            return self._row_to_entry(row)

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        return MemoryEntry(
            id=row["id"],
            repository=row["repository"],
            branch=row["branch"],
            key=row["key"],
            value=json.loads(row["value"]),
            context=json.loads(row["context"]) if row["context"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            accessed_at=datetime.fromisoformat(row["accessed_at"]),
        )

    def delete(
        self,
        key: str,
        repository: str = "default",
        branch: str = "main",
    ) -> bool:
        """Delete a memory entry.

        Args:
            key: Memory key
            repository: Repository scope
            branch: Branch scope

        Returns:
            True if entry was deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE repository = ? AND branch = ? AND key = ?",
                (repository, branch, key),
            )
            conn.commit()
            return cursor.rowcount > 0

    def search(
        self,
        pattern: str,
        repository: str = "default",
        branch: str = "main",
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Search memories by key pattern.

        Args:
            pattern: SQL LIKE pattern (use % for wildcard)
            repository: Repository scope
            branch: Branch scope
            limit: Maximum results

        Returns:
            List of matching MemoryEntry objects
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM memories
                WHERE repository = ? AND branch = ? AND key LIKE ?
                ORDER BY accessed_at DESC
                LIMIT ?
                """,
                (repository, branch, pattern, limit),
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]

    def list_keys(
        self,
        repository: str = "default",
        branch: str = "main",
        limit: int = 100,
    ) -> list[str]:
        """List all keys for a repository/branch.

        Args:
            repository: Repository scope
            branch: Branch scope
            limit: Maximum results

        Returns:
            List of keys
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT key FROM memories
                WHERE repository = ? AND branch = ?
                ORDER BY accessed_at DESC
                LIMIT ?
                """,
                (repository, branch, limit),
            )
            return [row["key"] for row in cursor.fetchall()]

    def get_recent(
        self,
        repository: str = "default",
        branch: str = "main",
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Get most recently accessed memories.

        Args:
            repository: Repository scope
            branch: Branch scope
            limit: Maximum results

        Returns:
            List of MemoryEntry objects ordered by access time
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM memories
                WHERE repository = ? AND branch = ?
                ORDER BY accessed_at DESC
                LIMIT ?
                """,
                (repository, branch, limit),
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]

    def get_all(
        self,
        repository: str = "default",
        branch: str = "main",
    ) -> dict[str, Any]:
        """Get all memories as a key-value dict.

        Args:
            repository: Repository scope
            branch: Branch scope

        Returns:
            Dict mapping keys to values
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT key, value FROM memories WHERE repository = ? AND branch = ?",
                (repository, branch),
            )
            return {row["key"]: json.loads(row["value"]) for row in cursor.fetchall()}

    def clear(
        self,
        repository: str | None = None,
        branch: str | None = None,
    ) -> int:
        """Clear memories.

        Args:
            repository: If provided, only clear this repository
            branch: If provided with repository, only clear this branch

        Returns:
            Number of entries deleted
        """
        with self._get_connection() as conn:
            if repository is None:
                # Clear all
                cursor = conn.execute("DELETE FROM memories")
            elif branch is None:
                # Clear repository
                cursor = conn.execute(
                    "DELETE FROM memories WHERE repository = ?",
                    (repository,),
                )
            else:
                # Clear specific branch
                cursor = conn.execute(
                    "DELETE FROM memories WHERE repository = ? AND branch = ?",
                    (repository, branch),
                )
            conn.commit()
            return cursor.rowcount

    def count(
        self,
        repository: str | None = None,
        branch: str | None = None,
    ) -> int:
        """Count memory entries.

        Args:
            repository: If provided, only count this repository
            branch: If provided with repository, only count this branch

        Returns:
            Number of entries
        """
        with self._get_connection() as conn:
            if repository is None:
                cursor = conn.execute("SELECT COUNT(*) as cnt FROM memories")
            elif branch is None:
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM memories WHERE repository = ?",
                    (repository,),
                )
            else:
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM memories WHERE repository = ? AND branch = ?",
                    (repository, branch),
                )
            count: int = int(cursor.fetchone()["cnt"])
            return count

    def list_repositories(self) -> list[str]:
        """List all repositories with stored memories.

        Returns:
            List of repository names
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT DISTINCT repository FROM memories ORDER BY repository")
            return [row["repository"] for row in cursor.fetchall()]

    def list_branches(self, repository: str) -> list[str]:
        """List all branches for a repository.

        Args:
            repository: Repository name

        Returns:
            List of branch names
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT branch FROM memories WHERE repository = ? ORDER BY branch",
                (repository,),
            )
            return [row["branch"] for row in cursor.fetchall()]

    def evict_lru(
        self,
        repository: str = "default",
        branch: str = "main",
        keep_count: int = 100,
    ) -> int:
        """Evict least recently used entries.

        Keeps the most recently accessed entries up to keep_count.

        Args:
            repository: Repository scope
            branch: Branch scope
            keep_count: Number of entries to keep

        Returns:
            Number of entries evicted
        """
        with self._get_connection() as conn:
            # Get IDs to keep
            cursor = conn.execute(
                """
                SELECT id FROM memories
                WHERE repository = ? AND branch = ?
                ORDER BY accessed_at DESC
                LIMIT ?
                """,
                (repository, branch, keep_count),
            )
            keep_ids = [row["id"] for row in cursor.fetchall()]

            if not keep_ids:
                return 0

            # Delete everything else (placeholders is a generated "?," list;
            # all values are parameterized — not user-interpolated SQL)
            placeholders = ",".join("?" * len(keep_ids))
            cursor = conn.execute(
                f"""
                DELETE FROM memories
                WHERE repository = ? AND branch = ? AND id NOT IN ({placeholders})
                """,  # nosec B608
                (repository, branch, *keep_ids),
            )
            conn.commit()
            return cursor.rowcount


def get_memory_store(db_path: Path | str | None = None) -> MemoryStore:
    """Factory function to get a MemoryStore instance.

    Args:
        db_path: Optional database path override

    Returns:
        MemoryStore instance
    """
    return MemoryStore(db_path=db_path)
