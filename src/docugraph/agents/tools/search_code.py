"""Search code tool for the coding agent.

Provides code-specific search with language and structure awareness.
"""

from __future__ import annotations

from typing import Any


def search_code(
    query: str,
    language: str | None = None,
    top_k: int = 10,
    include_context: bool = True,
    _context: Any = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Search for code examples in indexed documentation.

    Args:
        query: Search query (can be natural language or code snippet)
        language: Filter by programming language (python, javascript, etc.)
        top_k: Number of results to return
        include_context: Include surrounding context
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        List of code results with code, path, language, and context
    """
    from docugraph.retrieval.hybrid_search import (
        HybridRetriever,
        HybridSearchConfig,
        SearchMode,
    )

    # Enhance query for code search
    code_query = query
    if language:
        code_query = f"{language} {query}"

    config = HybridSearchConfig(include_graph=False)
    retriever = HybridRetriever(config=config)

    results = retriever.search(
        query=code_query,
        top_k=top_k * 2,  # Get more results for filtering
        mode=SearchMode.HYBRID,
    )

    # Filter and format code results
    formatted = []
    for result in results:
        chunk = result.chunk
        content = chunk.content

        # Skip if doesn't look like code
        if not _contains_code(content):
            continue

        # Filter by language if specified
        if language:
            detected_lang = _detect_language(content)
            if detected_lang and language.lower() not in detected_lang.lower():
                continue

        # Extract code blocks
        code_blocks = _extract_code_blocks(content)
        if not code_blocks:
            # Use full content if no blocks found
            code_blocks = [{"code": content, "language": _detect_language(content) or ""}]

        for block in code_blocks:
            formatted.append(
                {
                    "code": block["code"],
                    "language": block.get("language", ""),
                    "path": chunk.metadata.get("source_path")
                    or chunk.metadata.get("source_url", "Unknown"),
                    "title": chunk.metadata.get("title", ""),
                    "score": result.score,
                    "context": chunk.content if include_context else "",
                    "metadata": chunk.metadata,
                }
            )

        if len(formatted) >= top_k:
            break

    return formatted[:top_k]


def search_symbols(
    symbol_name: str,
    symbol_type: str | None = None,
    language: str | None = None,
    top_k: int = 10,
    _context: Any = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Search for code symbols (functions, classes, methods).

    Args:
        symbol_name: Name of the symbol to search for
        symbol_type: Type filter (function, class, method, variable)
        language: Programming language filter
        top_k: Number of results to return
        _context: Agent execution context (injected)
        **kwargs: Additional arguments

    Returns:
        List of symbol results
    """
    # Build query
    query_parts = [symbol_name]
    if symbol_type:
        query_parts.append(symbol_type)
    if language:
        query_parts.append(language)

    query = " ".join(query_parts)

    # Search with code focus
    results = search_code(
        query=query,
        language=language,
        top_k=top_k,
        include_context=True,
        _context=_context,
    )

    # Filter by symbol name presence
    filtered = []
    for r in results:
        code = r.get("code", "")
        if symbol_name.lower() in code.lower():
            filtered.append(r)

    return filtered[:top_k]


def _contains_code(content: str) -> bool:
    """Check if content likely contains code.

    Args:
        content: Text content to check

    Returns:
        True if content appears to contain code
    """
    # Check for code block markers
    if "```" in content or "~~~" in content:
        return True

    # Check for common code patterns
    code_indicators = [
        "def ",
        "class ",
        "function ",
        "import ",
        "from ",
        "const ",
        "let ",
        "var ",
        "async ",
        "await ",
        "return ",
        "=>",
        "() {",
        "(): ",
    ]

    return any(indicator in content for indicator in code_indicators)


def _detect_language(content: str) -> str | None:
    """Detect programming language from content.

    Args:
        content: Code content

    Returns:
        Detected language or None
    """
    # Check markdown code block language
    import re

    match = re.search(r"```(\w+)", content)
    if match:
        return match.group(1)

    # Heuristic detection
    if "def " in content and "import " in content:
        return "python"
    if "function " in content or "const " in content or "=>" in content:
        return "javascript"
    if "func " in content and "package " in content:
        return "go"
    if "fn " in content and "let " in content and "mut " in content:
        return "rust"
    if "public class " in content or "public static " in content:
        return "java"

    return None


def _extract_code_blocks(content: str) -> list[dict[str, str]]:
    """Extract code blocks from markdown content.

    Args:
        content: Markdown content

    Returns:
        List of code blocks with language and code
    """
    import re

    blocks = []

    # Match fenced code blocks
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)

    for lang, code in matches:
        blocks.append(
            {
                "language": lang or _detect_language(code) or "",
                "code": code.strip(),
            }
        )

    # Match indented code blocks (4 spaces)
    if not blocks:
        lines = content.split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            if line.startswith("    ") or line.startswith("\t"):
                code_lines.append(line[4:] if line.startswith("    ") else line[1:])
                in_code = True
            elif in_code and line.strip() == "":
                code_lines.append("")
            elif in_code:
                if code_lines:
                    blocks.append(
                        {
                            "language": _detect_language("\n".join(code_lines)) or "",
                            "code": "\n".join(code_lines).strip(),
                        }
                    )
                code_lines = []
                in_code = False

        if code_lines:
            blocks.append(
                {
                    "language": _detect_language("\n".join(code_lines)) or "",
                    "code": "\n".join(code_lines).strip(),
                }
            )

    return blocks
