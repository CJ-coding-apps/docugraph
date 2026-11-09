"""One-way migration from a Kùzu graph directory to a Ladybug database file.

The two engines cannot share a process. Their native modules both register
pybind types named ``Database``, and loading the second one fails with
``type "Database" is already registered``. So each engine gets a child process
that imports only itself, and each child asserts that the other engine is not
in ``sys.modules`` -- that assertion is the guard, not a comment.

The transfer is Kùzu's own ``EXPORT DATABASE`` (CSV) followed by Ladybug's
``IMPORT DATABASE``. Kùzu's export writes the full-text-search definitions it
created: a ``0_<TABLE>_FTS_TOKENIZE`` macro in ``schema.cypher`` (named with a
leading digit, and its regex literal contains a semicolon) and a
``CALL CREATE_FTS_INDEX(...)`` in ``index.cypher``. Measured on Ladybug 0.22.3
(P0-5): the macro itself parses, and an unstripped import with the FTS
extension loaded succeeds; without the extension the import fails at
``CREATE_FTS_INDEX`` with "function CREATE_FTS_INDEX is not defined". The
backend loads the extension itself only at initialisation, after the import,
so both statements are stripped first and the next Ladybug initialization
recreates the indexes (D-8).

Verification is per-table row counts, read in each engine. On any mismatch the
partial ``.lbdb`` is removed and nothing is left in a half-migrated state. The
Kùzu graph is never modified, and the configured backend is never changed: the
user decides when to switch.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

NODE_TABLES = ("Episodic", "Entity", "Community", "RelatesToNode_", "Saga")
REL_TABLES = ("RELATES_TO", "MENTIONS", "HAS_MEMBER", "HAS_EPISODE", "NEXT_EPISODE")


class MigrationError(Exception):
    """The graph could not be migrated; the message says what was left behind."""


_COUNTS = """
NODE_TABLES = %(node_tables)r
REL_TABLES = %(rel_tables)r
counts = {}
for table in NODE_TABLES:
    counts[table] = list(
        conn.execute(f"MATCH (n:{table}) RETURN count(n) AS n;").rows_as_dict()
    )[0]["n"]
for table in REL_TABLES:
    counts[table] = list(
        conn.execute(f"MATCH ()-[r:{table}]->() RETURN count(r) AS n;").rows_as_dict()
    )[0]["n"]
print(json.dumps(counts))
"""

_EXPORT_SCRIPT = (
    """
import json
import sys

import kuzu

assert "ladybug" not in sys.modules, "the export child loaded ladybug"

db = kuzu.Database(%(db_path)r)
conn = kuzu.Connection(db)
conn.execute("EXPORT DATABASE %(export_dir)r (format='csv');")
"""
    + _COUNTS
    + """
conn.close()
db.close()
"""
)

_IMPORT_SCRIPT = (
    """
import json
import sys

import ladybug

assert "kuzu" not in sys.modules, "the import child loaded kuzu"

db = ladybug.Database(%(db_path)r)
conn = ladybug.Connection(db)
conn.execute("IMPORT DATABASE %(export_dir)r")
"""
    + _COUNTS
    + """
conn.close()
db.close()
"""
)


def _counts_script_params() -> dict[str, Any]:
    return {"node_tables": NODE_TABLES, "rel_tables": REL_TABLES}


def _run_child(script: str, **params: Any) -> dict[str, int]:
    params.update(_counts_script_params())
    proc = subprocess.run(
        [sys.executable, "-c", script % params],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise MigrationError(
            "The graph migration step failed: "
            + (detail[-1] if detail else f"exit code {proc.returncode}")
        )
    lines = proc.stdout.strip().splitlines()
    if not lines:
        raise MigrationError("The graph migration step produced no result.")
    try:
        counts = json.loads(lines[-1])
    except ValueError as exc:
        raise MigrationError(
            f"The graph migration step produced unreadable output: {lines[-1]}"
        ) from exc
    return {str(k): int(v) for k, v in counts.items()}


def _run_export(db_path: Path, export_dir: Path) -> dict[str, int]:
    return _run_child(_EXPORT_SCRIPT, db_path=str(db_path), export_dir=str(export_dir))


def _run_import(export_dir: Path, db_path: Path) -> dict[str, int]:
    return _run_child(_IMPORT_SCRIPT, db_path=str(db_path), export_dir=str(export_dir))


def _split_statements(text: str) -> list[str]:
    """Split Cypher on top-level semicolons, keeping quoted text intact.

    A naive ``text.split(";")`` cuts Kùzu's generated tokenizer macro in half:
    its regex literal contains a ``;`` inside the quotes. The splitter tracks
    the quote character, honours backslash escapes and a doubled quote, and
    never splits inside a literal. Kùzu emits both forms: the regex itself
    contains ``\\'``, so ignoring backslash escapes re-opens the literal early
    and leaves exactly the fragment that failed to parse.
    """
    statements: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote is not None:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(text):
                buf.append(text[i + 1])
                i += 1
            elif ch == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    buf.append(text[i + 1])
                    i += 1
                else:
                    quote = None
        elif ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch == ";":
            statements.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf)
    if tail.strip():
        statements.append(tail)
    return statements


# Kùzu's export names the generated tokenizer macros after the index, e.g.
# `0_EPISODE_CONTENT_TOKENIZE`; the P0-5 probe in the WP used a name carrying
# "FTS" instead. Match the construct, not one spelling: a CREATE MACRO whose
# body tokenizes, plus the index statements themselves.
_FTS_STATEMENT = re.compile(
    r"(?:CREATE|DROP)_FTS_INDEX|\bCREATE\s+MACRO\b.*?TOKENIZE",
    re.IGNORECASE | re.DOTALL,
)


def _is_fts_definition(statement: str) -> bool:
    return _FTS_STATEMENT.search(statement) is not None


def strip_fts_definitions(export_dir: Path) -> int:
    """Remove full-text-search statements from a Kùzu export, in place.

    Returns the number of statements removed. ``index.cypher`` is rewritten to
    an empty file rather than deleted: the import reads a fixed set of files.
    """
    removed = 0
    for name in ("schema.cypher", "index.cypher"):
        path = export_dir / name
        if not path.is_file():
            continue
        text = path.read_text()
        if not text.strip():
            path.write_text("")
            continue
        kept: list[str] = []
        for statement in _split_statements(text):
            if not statement.strip():
                continue
            if _is_fts_definition(statement):
                removed += 1
            else:
                kept.append(statement.strip())
        path.write_text(";\n".join(kept) + (";\n" if kept else ""))
    return removed


def _remove_partial(path: Path) -> None:
    for suffix in ("", ".wal", ".lock"):
        Path(str(path) + suffix).unlink(missing_ok=True)


def migrate_graph(
    data_dir: Path,
    kuzu_path: Path | None = None,
    lbdb_path: Path | None = None,
) -> dict[str, int]:
    """Migrate ``graphiti.kuzu`` to ``graphiti.lbdb`` under ``data_dir/graph``.

    Refuses to run when the Kùzu graph is absent or the Ladybug file already
    exists. Returns the verified per-table counts.
    """
    graph_dir = Path(data_dir) / "graph"
    kuzu_path = Path(kuzu_path) if kuzu_path is not None else graph_dir / "graphiti.kuzu"
    lbdb_path = Path(lbdb_path) if lbdb_path is not None else graph_dir / "graphiti.lbdb"

    if not kuzu_path.exists():
        raise MigrationError(
            f"No Kùzu graph was found at {kuzu_path}; there is nothing to migrate."
        )
    if lbdb_path.exists():
        raise MigrationError(
            f"A Ladybug database already exists at {lbdb_path}; refusing to overwrite it. "
            "Move it away first if you want to migrate again."
        )

    with tempfile.TemporaryDirectory(prefix="docugraph-migrate-") as scratch:
        export_dir = Path(scratch) / "export"
        before = _run_export(kuzu_path, export_dir)
        strip_fts_definitions(export_dir)
        try:
            after = _run_import(export_dir, lbdb_path)
        except Exception:
            _remove_partial(lbdb_path)
            raise

    if after != before:
        _remove_partial(lbdb_path)
        differences = ", ".join(
            f"{table}: {before.get(table, 0)} -> {after.get(table, 0)}"
            for table in sorted(set(before) | set(after))
            if before.get(table, 0) != after.get(table, 0)
        )
        raise MigrationError(
            f"The migrated graph does not match the source ({differences}). "
            f"The partially imported database was removed; {kuzu_path} is unchanged."
        )

    return after
