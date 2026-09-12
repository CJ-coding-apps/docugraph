"""Memory operations tool for the coding agent.

Provides session memory for storing and retrieving context, decisions,
and other information scoped by repository and branch.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def memory_store(
    key: str,
    value: Any,
    repository: str | None = None,
    branch: str | None = None,
    _context: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Store a value in agent memory.

    Args:
        key: Key to store the value under
        value: Value to store (can be any JSON-serializable type)
        repository: Repository scope (defaults to context or 'default')
        branch: Branch scope (defaults to context or 'main')
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        Confirmation with stored entry details
    """
    from docugraph.storage.memory_store import MemoryStore

    # Get scope from context or defaults
    if _context:
        repository = repository or getattr(_context, "repository", "default")
        branch = branch or getattr(_context, "branch", "main")
    else:
        repository = repository or "default"
        branch = branch or "main"

    store = MemoryStore()
    entry = store.set(
        key=key,
        value=value,
        repository=repository,
        branch=branch,
    )

    return {
        "key": entry.key,
        "value": entry.value,
        "repository": entry.repository,
        "branch": entry.branch,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
    }


def memory_recall(
    key: str,
    repository: str | None = None,
    branch: str | None = None,
    default: Any = None,
    _context: Any = None,
    **kwargs: Any,
) -> Any:
    """Retrieve a value from agent memory.

    Args:
        key: Key to retrieve (or '*' for all keys)
        repository: Repository scope (defaults to context or 'default')
        branch: Branch scope (defaults to context or 'main')
        default: Default value if key not found
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        Stored value or default if not found
    """
    from docugraph.storage.memory_store import MemoryStore

    # Get scope from context or defaults
    if _context:
        repository = repository or getattr(_context, "repository", "default")
        branch = branch or getattr(_context, "branch", "main")
    else:
        repository = repository or "default"
        branch = branch or "main"

    store = MemoryStore()

    # List all keys
    if key == "*":
        keys = store.list_keys(repository=repository, branch=branch)
        return {
            "keys": keys,
            "repository": repository,
            "branch": branch,
            "total": len(keys),
        }

    # Get specific key
    value = store.get(key, repository=repository, branch=branch)
    if value is None:
        return default

    return value


def memory_delete(
    key: str,
    repository: str | None = None,
    branch: str | None = None,
    _context: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Delete a value from agent memory.

    Args:
        key: Key to delete
        repository: Repository scope
        branch: Branch scope
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        Confirmation with deletion status
    """
    from docugraph.storage.memory_store import MemoryStore

    # Get scope from context or defaults
    if _context:
        repository = repository or getattr(_context, "repository", "default")
        branch = branch or getattr(_context, "branch", "main")
    else:
        repository = repository or "default"
        branch = branch or "main"

    store = MemoryStore()
    deleted = store.delete(key, repository=repository, branch=branch)

    return {
        "key": key,
        "deleted": deleted,
        "repository": repository,
        "branch": branch,
    }


def memory_search(
    pattern: str,
    repository: str | None = None,
    branch: str | None = None,
    _context: Any = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Search memory keys by pattern.

    Args:
        pattern: Pattern to search for (supports % wildcard)
        repository: Repository scope
        branch: Branch scope
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        List of matching entries with keys and values
    """
    from docugraph.storage.memory_store import MemoryStore

    # Get scope from context or defaults
    if _context:
        repository = repository or getattr(_context, "repository", "default")
        branch = branch or getattr(_context, "branch", "main")
    else:
        repository = repository or "default"
        branch = branch or "main"

    store = MemoryStore()
    keys = store.list_keys(repository=repository, branch=branch)

    # Filter by pattern
    import fnmatch

    # Convert SQL-style wildcard to glob-style
    glob_pattern = pattern.replace("%", "*")
    matching_keys = fnmatch.filter(keys, glob_pattern)

    # Get values for matching keys
    results = []
    for key in matching_keys:
        value = store.get(key, repository=repository, branch=branch)
        results.append(
            {
                "key": key,
                "value": value,
                "repository": repository,
                "branch": branch,
            }
        )

    return results


def store_decision(
    decision: str,
    reasoning: str,
    context: dict[str, Any] | None = None,
    _context: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Store a decision with reasoning for future reference.

    Args:
        decision: The decision made
        reasoning: Why this decision was made
        context: Additional context
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        Stored decision entry
    """
    # Generate a unique key for the decision
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    key = f"decision_{timestamp}"

    value = {
        "decision": decision,
        "reasoning": reasoning,
        "context": context or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return memory_store(key=key, value=value, _context=_context)


def recall_decisions(
    limit: int = 10,
    _context: Any = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Recall recent decisions.

    Args:
        limit: Maximum number of decisions to return
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        List of recent decisions
    """
    results = memory_search(pattern="decision_%", _context=_context)

    # Sort by key (which includes timestamp) and return most recent
    results.sort(key=lambda x: x["key"], reverse=True)

    decisions = []
    for r in results[:limit]:
        value = r.get("value", {})
        if isinstance(value, dict):
            decisions.append(
                {
                    "key": r["key"],
                    "decision": value.get("decision"),
                    "reasoning": value.get("reasoning"),
                    "timestamp": value.get("timestamp"),
                }
            )

    return decisions


def store_context(
    context_type: str,
    data: dict[str, Any],
    _context: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Store contextual information.

    Args:
        context_type: Type of context (e.g., 'architecture', 'dependencies')
        data: Context data
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        Stored context entry
    """
    key = f"context_{context_type}"
    return memory_store(key=key, value=data, _context=_context)


def recall_context(
    context_type: str,
    _context: Any = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Recall contextual information.

    Args:
        context_type: Type of context to recall
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        Context data or None if not found
    """
    key = f"context_{context_type}"
    return memory_recall(key=key, default=None, _context=_context)
