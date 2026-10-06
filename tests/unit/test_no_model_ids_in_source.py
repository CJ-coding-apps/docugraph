"""No model id is compiled into the package.

A model id in the source is a decision the user did not make. It picks a vendor,
a price and a capability envelope at install time, and it goes stale silently:
the name keeps resolving to something until the day the provider retires it, at
which point every user of the package gets a 404 for a model they never chose.
Both cloud providers had one (`gpt-4o` in the LLM config, a table of OpenAI
embedding dimensions, `rerank-english-v3.0` on the Cohere reranker), and all have
been removed -- a cloud model is now named by the user and the call fails, saying
which setting to set, when it is not.

This test is the part that keeps them gone. It reads every string literal the
interpreter can evaluate, because that is exactly the set a request could be
built from: a name inside a docstring is prose a reader meets, but a name in a
constant, a default argument or an f-string is a name that can reach a socket.

The scan is deliberately blind to docstrings and comments -- the modules that
removed these names explain the removal by naming them, and that explanation is
worth more than the tidiness of a clean grep. Neither can reach a socket. It is
equally blind to `tests/` and `docs/`, where examples belong; the requirement
there is that they be *labelled* as examples, which is a judgement about prose
that a scanner cannot make.

The exceptions are local backends. A local model name is not a decision made on
the user's behalf: it is one they can `ollama pull` or let fastembed download,
checkable without an account, and the backend is unreachable unless they ask for
it. They are exempted by the *matched text*, not by the file, so a second model
id in the same file -- or in the same string -- still fails this test.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "docugraph"

# Model ids, by the shapes their vendors use. Written as patterns rather than a
# list of names because the defect is "a model name the user did not choose",
# and a list of names only catches the ones already known -- the next
# `gpt-4.1-mini` is the one nobody has heard of yet.
MODEL_ID_PATTERNS = {
    # `gpt-4o`, `gpt-4.1-mini`, `gpt-3.5-turbo`; and `o1-`, `o3-mini`, `o4-mini`.
    "openai-chat": r"\bgpt-[0-9][0-9a-z.\-]*",
    "openai-chatgpt": r"\bchatgpt-[0-9][0-9a-z.\-]*",
    "openai-reasoning": r"\bo[1-9][0-9]*(-[a-z0-9\-]+)?\b",
    "openai-embedding": r"\btext-embedding-[a-z0-9.\-]+",
    # `claude-3-haiku-20240307`, but also `claude-sonnet-4-20250514`: the family
    # name comes before the number, so a pattern requiring a digit straight after
    # the dash sees none of the current generation.
    "anthropic": r"\bclaude-[0-9a-z][0-9a-z.\-]*",
    # `command-r-plus`, `command-r7b`, and `command-light`; deliberately not
    # `command-[a-z0-9]*`, which would match the phrase "command-line".
    "cohere-command": r"\bcommand-(r|light|nightly|a|b)[0-9a-z.\-]*",
    "cohere-embed": r"\bembed-(english|multilingual)-v[0-9.]+",
    "cohere-rerank": r"\brerank-(english|multilingual)-v[0-9.]+",
    # Local names. These are the ones the exceptions below cover; a new one, or
    # the same one in a new file, is still a hit.
    "llama": r"\bllama[0-9][0-9a-z.\-]*",
    "local-tagged": r"\b(mistral|mixtral|qwen|gemma|deepseek|phi|nomic)[0-9a-z.\-]*:[0-9]",
    "local-embed": r"\b[a-z0-9]+(?:-[a-z0-9]+)*-embed-[a-z0-9.\-]+",
    "hf-repo": r"\b(BAAI|onnx-community|sentence-transformers|intfloat|nomic-ai)/[A-Za-z0-9.\-]+",
    "hf-minilm": r"\ball-MiniLM-[A-Za-z0-9.\-]+",
}

# Exceptions: (path relative to src/docugraph, the exact matched text) -> why.
ALLOWED: dict[tuple[str, str], str] = {
    ("core/llm.py", "llama3.2"): (
        "Ollama's default model. Ollama is local, needs no account, and the name "
        "is one `ollama pull llama3.2` away -- a fact the preflight error prints."
    ),
    ("core/embeddings.py", "BAAI/bge-small-en-v1.5"): (
        "fastembed's default. A local ONNX download, ~130 MB, no account."
    ),
    ("core/embeddings.py", "all-MiniLM-L6-v2"): (
        "sentence-transformers' default. Local, and only reached when the user names that backend."
    ),
    ("core/embeddings.py", "nomic-embed-text"): (
        "Ollama's embedding model. Local, pulled by the user, same as llama3.2."
    ),
    ("retrieval/reranker.py", "BAAI/bge-reranker-v2-m3"): (
        "The local cross-encoder's default, downloaded on first use by the "
        "opt-in --rerank flag. Reranking defaults to off."
    ),
    ("retrieval/reranker.py", "onnx-community/bge-reranker-v2-m3-ONNX"): (
        "The HuggingFace repository holding those weights, for that same opt-in local reranker."
    ),
    # Descriptions rather than defaults: the CLI flag and the MCP tool both have
    # to tell the user which model the 1.8 GB download is for, and neither
    # interface module can import the constant without pulling pydantic into
    # every CLI invocation (~320 ms, measured).
    ("interfaces/cli.py", "BAAI/bge-reranker-v2-m3"): (
        "Help text for `--rerank`, naming the model it downloads. Held to the "
        "constant by the test at the bottom of this file."
    ),
    ("interfaces/mcp_server.py", "BAAI/bge-reranker-v2-m3"): (
        "The same name in the MCP tool description, for the same reason."
    ),
}

# The two exceptions above describe a default held elsewhere, so unlike the
# others they must keep agreeing with it: a rename that forgot the help text
# would leave the user expecting a download that does not happen.
_RERANKER_NAME_REPEATED_IN_HELP = ["interfaces/cli.py", "interfaces/mcp_server.py"]


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Ids of the constants that are docstrings rather than live strings."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return found


def _string_literals(path: Path) -> list[tuple[int, str]]:
    """Every evaluatable string literal in `path`, with its line number.

    Adjacent literals are one `ast.Constant` holding their concatenation, which
    is what the code would send, so that is what is checked.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = _docstring_nodes(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _hits(allow: dict[tuple[str, str], str]) -> list[tuple[str, str, str]]:
    """Every (path, matched text, description) a model id was found at.

    Matching is per model id rather than per literal, so exempting one name
    cannot excuse a different one that shares its string.
    """
    compiled = {name: re.compile(pattern) for name, pattern in MODEL_ID_PATTERNS.items()}
    hits: list[tuple[str, str, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        for lineno, text in _string_literals(path):
            for name, pattern in compiled.items():
                for match in pattern.finditer(text):
                    if (relative, match.group(0)) in allow:
                        continue
                    hits.append(
                        (
                            f"{relative}:{lineno}",
                            match.group(0),
                            f"{name}: {match.group(0)!r} in {text[:70]!r}",
                        )
                    )
    return hits


def test_no_model_id_is_compiled_into_the_source() -> None:
    hits = _hits(ALLOWED)
    assert not hits, (
        "Model ids belong in configuration, not in the package. A cloud model "
        "must be named by the user; a local default must be added to ALLOWED "
        "with the reason it is safe.\n  " + "\n  ".join(description for _, _, description in hits)
    )


@pytest.mark.parametrize(("path", "text"), sorted(ALLOWED))
def test_every_exception_is_still_doing_work(path: str, text: str) -> None:
    """An exception that matches nothing any more is a stale exemption.

    Measured by re-running the scan with the exception withheld: if the hit does
    not come back, nothing was being suppressed, and the entry has outlived the
    code it was written for -- at which point it is only widening the test.
    """
    unexempted = {(hit_path.split(":")[0], hit_text) for hit_path, hit_text, _ in _hits({})}
    assert (path, text) in unexempted, (
        f"{path} has no occurrence of {text!r} to exempt, so its entry in "
        "ALLOWED is stale and should be removed"
    )


def test_the_name_repeated_in_help_text_is_still_the_reranker_default() -> None:
    """The exceptions that are a copy of a value are held to that value."""
    from docugraph.retrieval.reranker import FastembedReranker

    for path in _RERANKER_NAME_REPEATED_IN_HELP:
        literals = [text for _, text in _string_literals(SRC / path)]
        assert any(FastembedReranker.DEFAULT_MODEL in text for text in literals), (
            f"{path} does not name FastembedReranker.DEFAULT_MODEL "
            f"({FastembedReranker.DEFAULT_MODEL!r}); its help text would promise "
            "a download that does not happen"
        )
