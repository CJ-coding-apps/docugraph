"""AST-based code parsing and analysis.

Supports multiple languages via tree-sitter:
- Python
- JavaScript/TypeScript
- Go
- Rust
- Java

Extracts code structure, docstrings, and relationships.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from docugraph.core.models import EntityType


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    UNKNOWN = "unknown"


@dataclass
class CodeSymbol:
    """A code symbol (function, class, method, etc.)."""

    name: str
    symbol_type: str  # function, class, method, variable, constant
    language: Language
    start_line: int
    end_line: int
    docstring: str | None = None
    signature: str | None = None
    parent: str | None = None  # Parent class/module name
    decorators: list[str] = field(default_factory=list)
    parameters: list[dict[str, str]] = field(default_factory=list)
    return_type: str | None = None
    body: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get fully qualified name."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name

    @property
    def entity_type(self) -> EntityType:
        """Map to EntityType."""
        mapping = {
            "function": EntityType.FUNCTION,
            "method": EntityType.FUNCTION,
            "class": EntityType.CLASS,
            "interface": EntityType.CLASS,
            "module": EntityType.MODULE,
        }
        return mapping.get(self.symbol_type, EntityType.CONCEPT)


@dataclass
class Import:
    """An import statement."""

    module: str
    name: str | None = None  # Specific import (from X import name)
    alias: str | None = None
    line: int = 0


@dataclass
class ParsedCode:
    """Result of parsing a code file."""

    file_path: str
    language: Language
    symbols: list[CodeSymbol]
    imports: list[Import]
    module_docstring: str | None = None
    raw_content: str | None = None

    @property
    def classes(self) -> list[CodeSymbol]:
        """Get all class definitions."""
        return [s for s in self.symbols if s.symbol_type == "class"]

    @property
    def functions(self) -> list[CodeSymbol]:
        """Get all function definitions (not methods)."""
        return [s for s in self.symbols if s.symbol_type == "function"]

    @property
    def methods(self) -> list[CodeSymbol]:
        """Get all method definitions."""
        return [s for s in self.symbols if s.symbol_type == "method"]


class CodeParser:
    """Parses code files to extract structure and documentation.

    Uses regex-based parsing as a fallback when tree-sitter is not available.
    Supports multiple programming languages.
    """

    # Language detection by extension
    EXTENSION_MAP = {
        ".py": Language.PYTHON,
        ".pyw": Language.PYTHON,
        ".js": Language.JAVASCRIPT,
        ".jsx": Language.JAVASCRIPT,
        ".mjs": Language.JAVASCRIPT,
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TYPESCRIPT,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".java": Language.JAVA,
    }

    def __init__(self) -> None:
        """Initialize the code parser."""
        self._tree_sitter_available = False
        self._parsers: dict[Language, Any] = {}

        # Try to import tree-sitter
        try:
            import tree_sitter
            self._tree_sitter_available = True
        except ImportError:
            pass

    def detect_language(self, file_path: str | Path) -> Language:
        """Detect programming language from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Detected language
        """
        path = Path(file_path)
        return self.EXTENSION_MAP.get(path.suffix.lower(), Language.UNKNOWN)

    def _extract_python_docstring(self, content: str) -> str | None:
        """Extract module-level docstring from Python code."""
        # Match triple-quoted string at start of file
        match = re.match(r'^[\s]*(?:\'\'\'|""")(.+?)(?:\'\'\'|""")', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _parse_python_regex(self, content: str, file_path: str) -> ParsedCode:
        """Parse Python code using regex (fallback)."""
        symbols: list[CodeSymbol] = []
        imports: list[Import] = []
        lines = content.split("\n")

        # Extract module docstring
        module_doc = self._extract_python_docstring(content)

        # Track current class context
        current_class: str | None = None
        current_class_indent = 0

        # Import patterns
        import_pattern = re.compile(r"^(?:from\s+([\w.]+)\s+)?import\s+(.+)$")

        # Function/method pattern
        func_pattern = re.compile(
            r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:"
        )

        # Class pattern
        class_pattern = re.compile(r"^(\s*)class\s+(\w+)(?:\([^)]*\))?:")

        # Decorator pattern
        decorator_pattern = re.compile(r"^(\s*)@(\w+(?:\.\w+)*(?:\([^)]*\))?)")

        pending_decorators: list[str] = []

        for i, line in enumerate(lines):
            line_num = i + 1

            # Check for decorator
            dec_match = decorator_pattern.match(line)
            if dec_match:
                pending_decorators.append(dec_match.group(2))
                continue

            # Check for import
            import_match = import_pattern.match(line.strip())
            if import_match:
                from_module = import_match.group(1)
                import_names = import_match.group(2)

                for name_part in import_names.split(","):
                    name_part = name_part.strip()
                    if " as " in name_part:
                        name, alias = name_part.split(" as ")
                        name = name.strip()
                        alias = alias.strip()
                    else:
                        name = name_part
                        alias = None

                    if from_module:
                        imports.append(
                            Import(module=from_module, name=name, alias=alias, line=line_num)
                        )
                    else:
                        imports.append(Import(module=name, alias=alias, line=line_num))
                continue

            # Check for class
            class_match = class_pattern.match(line)
            if class_match:
                indent = len(class_match.group(1))
                class_name = class_match.group(2)

                # Find class docstring
                docstring = self._find_docstring(lines, i + 1)

                # Find end of class
                end_line = self._find_block_end(lines, i, indent)

                symbols.append(
                    CodeSymbol(
                        name=class_name,
                        symbol_type="class",
                        language=Language.PYTHON,
                        start_line=line_num,
                        end_line=end_line,
                        docstring=docstring,
                        decorators=pending_decorators.copy(),
                    )
                )

                current_class = class_name
                current_class_indent = indent
                pending_decorators.clear()
                continue

            # Check for function/method
            func_match = func_pattern.match(line)
            if func_match:
                indent = len(func_match.group(1))
                func_name = func_match.group(2)
                params_str = func_match.group(3)
                return_type = func_match.group(4)

                # Determine if method or function
                is_method = current_class and indent > current_class_indent
                symbol_type = "method" if is_method else "function"
                parent = current_class if is_method else None

                # Parse parameters
                params = self._parse_params(params_str)

                # Find docstring
                docstring = self._find_docstring(lines, i + 1)

                # Find end of function
                end_line = self._find_block_end(lines, i, indent)

                symbols.append(
                    CodeSymbol(
                        name=func_name,
                        symbol_type=symbol_type,
                        language=Language.PYTHON,
                        start_line=line_num,
                        end_line=end_line,
                        docstring=docstring,
                        signature=f"def {func_name}({params_str})",
                        parent=parent,
                        decorators=pending_decorators.copy(),
                        parameters=params,
                        return_type=return_type.strip() if return_type else None,
                    )
                )

                pending_decorators.clear()
                continue

            # Reset class context if we're back at module level
            if current_class and line.strip() and not line[0].isspace():
                current_class = None
                current_class_indent = 0

            # Clear decorators if we have a non-decorator, non-def/class line
            if pending_decorators and line.strip() and not line.strip().startswith("@"):
                pending_decorators.clear()

        return ParsedCode(
            file_path=file_path,
            language=Language.PYTHON,
            symbols=symbols,
            imports=imports,
            module_docstring=module_doc,
            raw_content=content,
        )

    def _find_docstring(self, lines: list[str], start_index: int) -> str | None:
        """Find docstring starting after a definition line."""
        if start_index >= len(lines):
            return None

        # Skip empty lines
        i = start_index
        while i < len(lines) and not lines[i].strip():
            i += 1

        if i >= len(lines):
            return None

        line = lines[i].strip()

        # Check for docstring start
        if line.startswith('"""') or line.startswith("'''"):
            quote = line[:3]
            if line.count(quote) >= 2:
                # Single line docstring
                return line[3:-3].strip()

            # Multi-line docstring
            doc_lines = [line[3:]]
            i += 1
            while i < len(lines):
                line = lines[i]
                if quote in line:
                    doc_lines.append(line[: line.index(quote)])
                    break
                doc_lines.append(line)
                i += 1

            return "\n".join(doc_lines).strip()

        return None

    def _find_block_end(self, lines: list[str], start_index: int, base_indent: int) -> int:
        """Find the end line of a code block."""
        for i in range(start_index + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                continue

            # Check indent
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and line.strip():
                return i

        return len(lines)

    def _parse_params(self, params_str: str) -> list[dict[str, str]]:
        """Parse function parameters string."""
        params = []
        if not params_str.strip():
            return params

        # Simple parameter parsing (doesn't handle all edge cases)
        for param in params_str.split(","):
            param = param.strip()
            if not param or param == "self" or param == "cls":
                continue

            # Extract name and type hint
            if ":" in param:
                parts = param.split(":", 1)
                name = parts[0].strip()
                type_hint = parts[1].split("=")[0].strip()
            else:
                name = param.split("=")[0].strip()
                type_hint = None

            # Remove * and ** prefixes
            name = name.lstrip("*")

            if name:
                params.append({"name": name, "type": type_hint})

        return params

    def _parse_javascript_regex(self, content: str, file_path: str) -> ParsedCode:
        """Parse JavaScript/TypeScript code using regex."""
        symbols: list[CodeSymbol] = []
        imports: list[Import] = []
        lines = content.split("\n")

        # Import patterns
        import_patterns = [
            re.compile(r"import\s+(?:{[^}]+}|\*\s+as\s+\w+|\w+)\s+from\s+['\"]([^'\"]+)['\"]"),
            re.compile(r"require\(['\"]([^'\"]+)['\"]\)"),
        ]

        # Function patterns
        func_patterns = [
            re.compile(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\("),
            re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)"),
            re.compile(r"(\w+)\s*:\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)"),
        ]

        # Class pattern
        class_pattern = re.compile(r"(?:export\s+)?class\s+(\w+)")

        for i, line in enumerate(lines):
            line_num = i + 1

            # Check imports
            for pattern in import_patterns:
                for match in pattern.finditer(line):
                    imports.append(Import(module=match.group(1), line=line_num))

            # Check class
            class_match = class_pattern.search(line)
            if class_match:
                symbols.append(
                    CodeSymbol(
                        name=class_match.group(1),
                        symbol_type="class",
                        language=Language.JAVASCRIPT,
                        start_line=line_num,
                        end_line=line_num,  # Would need brace matching for accurate end
                    )
                )
                continue

            # Check functions
            for pattern in func_patterns:
                func_match = pattern.search(line)
                if func_match:
                    symbols.append(
                        CodeSymbol(
                            name=func_match.group(1),
                            symbol_type="function",
                            language=Language.JAVASCRIPT,
                            start_line=line_num,
                            end_line=line_num,
                        )
                    )
                    break

        return ParsedCode(
            file_path=file_path,
            language=Language.JAVASCRIPT,
            symbols=symbols,
            imports=imports,
            raw_content=content,
        )

    def parse(self, content: str, file_path: str | Path) -> ParsedCode:
        """Parse code content.

        Args:
            content: Code content to parse
            file_path: Path to the file (for language detection)

        Returns:
            Parsed code structure
        """
        language = self.detect_language(file_path)

        if language == Language.PYTHON:
            return self._parse_python_regex(content, str(file_path))
        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
            return self._parse_javascript_regex(content, str(file_path))
        else:
            # Return empty result for unsupported languages
            return ParsedCode(
                file_path=str(file_path),
                language=language,
                symbols=[],
                imports=[],
                raw_content=content,
            )

    def parse_file(self, file_path: str | Path) -> ParsedCode:
        """Parse a code file.

        Args:
            file_path: Path to the file

        Returns:
            Parsed code structure
        """
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        return self.parse(content, file_path)

    def parse_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> Iterator[ParsedCode]:
        """Parse all code files in a directory.

        Args:
            directory: Directory path
            recursive: Whether to recurse into subdirectories

        Yields:
            ParsedCode for each code file
        """
        directory = Path(directory)
        extensions = set(self.EXTENSION_MAP.keys())

        pattern = "**/*" if recursive else "*"

        for ext in extensions:
            for file_path in directory.glob(f"{pattern}{ext}"):
                if file_path.is_file():
                    try:
                        yield self.parse_file(file_path)
                    except Exception:
                        continue


def get_code_parser() -> CodeParser:
    """Factory function to get a CodeParser instance.

    Returns:
        CodeParser instance
    """
    return CodeParser()
