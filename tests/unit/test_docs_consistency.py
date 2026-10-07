"""The pages describe the program that is installed, and this keeps them doing it.

Written before release, because a reader's first experience of a new package is
its documentation and a page that names a command, an environment variable or a
tool that does not exist is indistinguishable from a broken install.

Nothing here is a list copied out of the docs. Every claim is re-derived from
the object that owns it -- the pydantic settings model, the click command tree,
and the server's own `list_tools` -- so a rename has to be made in both places.

The same test exists in Cognitive-Fabric-v1, where the drift had accumulated
further: 22 parameters that no tool accepted, a `--limit` on a command that had
none, and a JSON log example keying on two fields the logger never wrote.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"

# The pages a reader is expected to follow. `claude-integration/` is included
# because it ships in the repository and its hooks run on a user's machine.
PAGES = sorted(
    [*DOCS_DIR.rglob("*.md"), ROOT / "README.md", *sorted((ROOT / "claude-integration").rglob("*.md"))]
)

# Shell the repository ships and people run. A command here that does not
# resolve is a hook that fails silently or noisily on every session.
SHELL_SOURCES = sorted(
    (ROOT / "claude-integration" / "hooks").glob("*.sh")
)

# Config files whose environment variables are part of the deployment surface.
ENV_SOURCES = [
    *PAGES,
    ROOT / "docker" / "Dockerfile",
    ROOT / "docker" / "docker-compose.yml",
    ROOT / "docker-compose.yml",
]

FENCED = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

# Variables the package reads but does not own, so they carry no settings
# field. Each is named here with the reader that consumes it, so the list is a
# statement about the code rather than a place to silence the check.
EXTERNAL_ENV = {
    "OPENAI_API_KEY": "read by LLMConfig.model_post_init",
    "ANTHROPIC_API_KEY": "read by LLMConfig.model_post_init",
    "COHERE_API_KEY": "read by the Cohere embedder",
    "FASTEMBED_CACHE_PATH": "read by the fastembed backend",
}

SCRIPT_NAMES = {"docugraph", "docugraph-mcp"}


def _fenced_blocks(page: Path):
    for match in FENCED.finditer(page.read_text()):
        yield match.group(1), match.group(2)


def _blocks(page: Path, *languages: str):
    for info, text in _fenced_blocks(page):
        if not languages or info in languages:
            yield text


# --------------------------------------------------------------------------
# 1. Environment variables
# --------------------------------------------------------------------------

ENV_ASSIGNMENT = re.compile(r"^\s*(?:export\s+|-)?([A-Z][A-Z0-9_]*)\s*=")


def _env_names() -> list[tuple[str, str]]:
    """(page, name) for every `DOCUGRAPH_*` variable a code block sets.

    Only the prefixed names: `IMG=$(docker compose build -q cli)` is a shell
    variable in an example, not configuration, and flagging it would be noise.
    A `DOCUGRAPH_` name, by contrast, is a claim that the program reads it.
    """
    found = []
    for page in ENV_SOURCES:
        relative = str(page.relative_to(ROOT))
        for text in _blocks(page):
            for line in text.splitlines():
                match = ENV_ASSIGNMENT.match(line)
                if match and match.group(1).startswith("DOCUGRAPH_"):
                    found.append((relative, match.group(1)))
    return found


def test_the_external_variable_allowlist_is_not_stale() -> None:
    """Every exemption is still a variable the repository actually names."""
    named = set()
    for page in ENV_SOURCES:
        named.update(re.findall(r"\b([A-Z][A-Z0-9_]{3,})\b", page.read_text()))
    unused = sorted(set(EXTERNAL_ENV) - named)
    assert not unused, f"EXTERNAL_ENV lists variables nothing names any more: {unused}"


def _settings() -> dict[str, set[str]]:
    """The real `DOCUGRAPH_<SECTION>__<KEY>` surface, from the settings model."""
    from docugraph.core.config import Config

    sections = {}
    for name, field in Config.model_fields.items():
        nested = getattr(field.annotation, "model_fields", None)
        if nested is None:
            continue
        sections[name.upper()] = set(nested)
    return sections


def test_the_scan_finds_environment_variables() -> None:
    # A floor, not a target: the pages currently name nine. What this guards
    # against is the scan silently returning nothing -- a broken regex, or a
    # page dropping out of ENV_SOURCES -- which would make the check below
    # vacuously true.
    assert len(_env_names()) >= 6


def test_the_scan_finds_the_settings_sections() -> None:
    assert set(_settings()) == {"STORAGE", "EMBEDDINGS", "LLM", "CRAWLER"}


def test_every_documented_environment_variable_is_a_settings_field() -> None:
    """A name in a page has to be one pydantic-settings will read.

    `extra="ignore"` means an unknown name is accepted by the model and then
    ignored, so a misspelled `DOCUGRAPH_STORAGE__DATA_DIRECTORY` leaves the
    reader's data where they did not ask for it, with nothing in the output to
    say so.
    """
    sections = _settings()
    problems: list[str] = []
    for page, name in _env_names():
        body = name[len("DOCUGRAPH_") :]
        if "__" not in body:
            problems.append(
                f"{page}: `{name}` needs the `__` nested delimiter "
                f"(e.g. DOCUGRAPH_STORAGE__DATA_DIR); the single-underscore form "
                "matches no field"
            )
            continue
        section, key = body.split("__", 1)
        if section not in sections:
            problems.append(
                f"{page}: `{name}` names no settings section "
                f"(have: {', '.join(sorted(sections))})"
            )
        elif key.lower() not in sections[section]:
            problems.append(
                f"{page}: `{name}` is not a field of {section.lower()} "
                f"(have: {', '.join(sorted(sections[section]))})"
            )
    assert not problems, "\n  ".join(
        ["Environment variables in the docs must be ones the code reads:"] + problems
    )


# --------------------------------------------------------------------------
# 2. MCP tool names
# --------------------------------------------------------------------------


def _advertised_tools() -> set[str]:
    """The names the server itself lists, not a copy kept in this file."""
    import asyncio

    from docugraph.interfaces import mcp_server

    return {tool.name for tool in asyncio.run(mcp_server.list_tools())}


# Two shapes assert a tool name, and both are checked because each is the
# place a reader would copy a name from.
#
# The first is a name immediately before the word "tool" -- what the skills
# say ("call the `hybrid_search` tool"). The underscore is what makes this
# precise: English prose has no `hybrid_search`, and every tool name here has
# one.
TOOL_MENTION = re.compile(r"`?\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`?\s+(?:MCP\s+)?tool", re.I)

# The second is the README's tool table, whose first cell is the name in
# backticks. This is the primary list and the earlier scan walked straight
# past it -- it only looked for the word "tool", which a table row does not
# contain, so a renamed row was invisible to it.
TOOL_TABLE_ROW = re.compile(r"^\|\s*`([a-z][a-z0-9_]+)`\s*\|", re.M)


def _tool_mentions() -> list[tuple[str, str]]:
    found = []
    for page in PAGES:
        relative = str(page.relative_to(ROOT))
        text = page.read_text()
        for match in TOOL_MENTION.finditer(text):
            found.append((relative, match.group(1)))
        for match in TOOL_TABLE_ROW.finditer(text):
            found.append((relative, match.group(1)))
    return found


def test_the_scan_finds_tool_mentions() -> None:
    assert len(_tool_mentions()) >= 5


def test_every_tool_a_page_names_is_one_the_server_advertises() -> None:
    """The skills tell an assistant which tool to call; a renamed tool breaks them."""
    advertised = _advertised_tools()
    problems = [
        f"{page}: names the `{name}` tool, which the server does not advertise "
        f"(it advertises {', '.join(sorted(advertised))})"
        for page, name in _tool_mentions()
        if name not in advertised
    ]
    assert not problems, "\n  ".join(
        ["Tools named in the docs must exist:"] + sorted(set(problems))
    )


def test_every_advertised_tool_is_documented() -> None:
    """And the reverse: the name a client shows is one a reader can find.

    A tool is called by its advertised name, not by its description, so a page
    that only paraphrases it ("Check stats") leaves the reader unable to tell
    which tool that is.
    """
    text = "\n".join(page.read_text() for page in PAGES)
    undocumented = [name for name in sorted(_advertised_tools()) if name not in text]
    assert not undocumented, (
        "These tools are advertised but named on no page a reader would look:\n  "
        + "\n  ".join(undocumented)
    )


# --------------------------------------------------------------------------
# 3. CLI commands
# --------------------------------------------------------------------------

# Shell keywords and prefixes that can precede the command in a line.
LEADING = {"if", "then", "else", "fi", "for", "do", "done", "while", "export",
           "set", "source", ".", "uv", "run", "python", "-m", "sudo", "env"}
ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
QUOTED = re.compile(r"^[\"']?(.*?)[\"']?$")


def _command_argv(line: str) -> list[str] | None:
    """The argv from the first console-script name on the line, or None.

    `command -v docugraph &> /dev/null` names the program without running it,
    so a redirection or operator right after the name means this line is a
    test, not an invocation.
    """
    line = line.split("#", 1)[0].strip().rstrip("\\").strip()
    if not line:
        return None
    tokens = [QUOTED.match(t).group(1) for t in line.split()]
    for index, token in enumerate(tokens):
        if token not in SCRIPT_NAMES:
            continue
        rest = tokens[index + 1 :]
        if rest and rest[0][:1] in {"&", ">", "<", "|", ";"}:
            return None
        return tokens[index:]
    return None


def _cli_invocations() -> list[tuple[str, str, list[str]]]:
    """(source, line, argv) for every line that runs a console script."""
    found = []
    for source in [*PAGES, *SHELL_SOURCES]:
        relative = str(source.relative_to(ROOT))
        blocks = _blocks(source, "bash", "sh", "shell", "console", "json") if source.suffix == ".md" else [source.read_text()]
        for text in blocks:
            for line in text.splitlines():
                argv = _command_argv(line)
                if argv:
                    found.append((relative, line.strip(), argv))
    return found


def _group_options() -> set[str]:
    from docugraph.interfaces.cli import main

    accepted: set[str] = set()
    for param in main.params:
        accepted.update(getattr(param, "opts", []))
        accepted.update(getattr(param, "secondary_opts", []))
    return accepted


def test_the_scan_finds_cli_invocations() -> None:
    assert len(_cli_invocations()) >= 15


def test_every_documented_command_exists() -> None:
    """Checked against the click tree, not against a list in this file."""
    import click

    from docugraph.interfaces.cli import main

    scripts = set(
        tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    )
    problems: list[str] = []
    for source, line, argv in _cli_invocations():
        name = argv[0]
        if name not in scripts:
            problems.append(
                f"{source}: `{name}` is not an installed command "
                f"(installed: {', '.join(sorted(scripts))})\n    {line}"
            )
            continue
        if name != "docugraph":
            continue

        # Descend the command tree: `index` is a group, so `index local`'s
        # options are not `index`'s. A positional that names a subcommand takes
        # us one level deeper; anything else is an argument.
        accepted = _group_options()
        node = main
        path = ["docugraph"]
        for token in argv[1:]:
            if token.startswith("-"):
                if token not in accepted:
                    problems.append(
                        f"{source}: `{' '.join(path)} {token}` is not an option of "
                        f"`{' '.join(path)}`\n    {line}"
                    )
                continue
            if isinstance(node, click.Group) and token in node.commands:
                node = node.commands[token]
                path.append(token)
                for param in node.params:
                    accepted.update(getattr(param, "opts", []))
                    accepted.update(getattr(param, "secondary_opts", []))
            elif path == ["docugraph"]:
                problems.append(
                    f"{source}: `docugraph {token}` is not a command "
                    f"(known: {', '.join(sorted(node.commands))})\n    {line}"
                )
                break
            # Otherwise it is a positional argument (a path, a URL, a query).
    assert not problems, "\n  ".join(
        ["Command lines in the repository must work:"] + problems
    )


def test_every_command_is_documented() -> None:
    """And the pages list them all, so a new one is not invisible.

    Matched with a trailing boundary: `docugraph configure` is not a mention of
    `docugraph config`, and a plain substring test would take it for one.
    """
    from docugraph.interfaces.cli import main

    text = "\n".join(page.read_text() for page in PAGES)
    undocumented = [
        name
        for name in sorted(main.commands)
        if not re.search(rf"docugraph {re.escape(name)}\b", text)
    ]
    assert not undocumented, (
        "These CLI commands appear on no page a reader would look at:\n  "
        + "\n  ".join(undocumented)
    )


# --------------------------------------------------------------------------
# 4. The fastembed cache default
# --------------------------------------------------------------------------

# Everything that states where fastembed caches. The source docstrings are
# included because they make the same claim to a different reader.
CACHE_CLAIM_SOURCES = [
    *PAGES,
    ROOT / "src" / "docugraph" / "core" / "embeddings.py",
    ROOT / "src" / "docugraph" / "retrieval" / "reranker.py",
]

# A page claiming that an unset FASTEMBED_CACHE_PATH means `~/.cache/fastembed`.
# The export example on the configuration page names that directory too, and is
# not a claim about the default, so the word "default" is required.
FALSE_DEFAULT = re.compile(r"default[^.]{0,200}?\.cache/fastembed")


def _claims_false_default(text: str) -> bool:
    """True if `text` calls `~/.cache/fastembed` the default.

    Whitespace is collapsed first: one of the offending claims was wrapped
    across two lines, and matching line by line walked straight past it.
    """
    return bool(FALSE_DEFAULT.search(re.sub(r"\s+", " ", text)))


def _fastembed_default_cache_dir() -> Path:
    """Where fastembed caches when `FASTEMBED_CACHE_PATH` is unset.

    Read under an environment with that variable removed: `ci.yml` sets it to
    the very path the pages used to claim was the default, so asking with the
    variable in place would agree with the pages for the wrong reason.
    """
    import os
    from unittest import mock

    from fastembed.common.utils import define_cache_dir

    without = {k: v for k, v in os.environ.items() if k != "FASTEMBED_CACHE_PATH"}
    with mock.patch.dict(os.environ, without, clear=True):
        return define_cache_dir()


def test_the_documented_fastembed_cache_default_is_the_real_one() -> None:
    """Four places said `~/.cache/fastembed`. That is not the default.

    fastembed's `define_cache_dir` falls back to `<tempdir>/fastembed_cache`.
    Nothing caught it because `ci.yml` pins `FASTEMBED_CACHE_PATH` to
    `~/.cache/fastembed`, so the guess was right in the one environment that
    ran the tests and wrong on every reader's machine.
    """
    import tempfile

    real = _fastembed_default_cache_dir()
    assert real == Path(tempfile.gettempdir()) / "fastembed_cache", (
        f"fastembed's default cache directory moved to {real}. The pages that "
        "describe where models are cached were written against the old one; "
        "re-read them."
    )
    assert real != Path.home() / ".cache" / "fastembed"

    offenders = [
        str(source.relative_to(ROOT))
        for source in CACHE_CLAIM_SOURCES
        if _claims_false_default(source.read_text())
    ]
    assert not offenders, (
        "These call `~/.cache/fastembed` the default, and it is not -- the "
        f"default is {real}:\n  " + "\n  ".join(offenders)
    )
