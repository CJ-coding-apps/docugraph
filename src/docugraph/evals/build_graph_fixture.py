"""Regenerate the committed graph fixture (WP §4.4, D-3).

``python -m docugraph.evals.build_graph_fixture --i-am-regenerating``

This is bench tooling, not a CLI command, and it is destructive in the only
sense that matters: it overwrites a committed artifact with the output of a
live language model. The three refusals are the contract:

1. Without ``--i-am-regenerating`` it exits non-zero and names the flag. There
   is no default output path and no "did you mean" prompt.
2. Without a configured, reachable LLM it exits non-zero and says an LLM (and
   its key/configuration) is required -- before it writes anything, so a refused
   run cannot leave a half-built fixture behind.
3. With uncommitted changes under ``src/`` it exits non-zero. The manifest
   records the commit that built the fixture (``extraction_git_sha``), so
   building from a working tree would make that record false.

The regeneration itself is not reproducible: ``deepseek-v4.1-flash:cloud`` is
served remotely and the same tag can mean different weights later. The manifest
says so. What *is* reproducible is the committed fixture, which the bench reads
without an LLM; this builder is the only path that talks to one.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from docugraph.core.embeddings import get_embedder
from docugraph.core.llm import get_llm_provider
from docugraph.evals.dataset import default_dataset_dir, load_corpus, sha256_file
from docugraph.evals.graph_bench import (
    FIXTURE_FORMAT,
    GRAPH_GROUP_ID,
    export_graph_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "retrieval" / "graph_fixture"
INGEST_REFERENCE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docugraph.evals.build_graph_fixture",
        description="Regenerate the committed graph fixture from the bench corpus.",
    )
    parser.add_argument(
        "--i-am-regenerating",
        action="store_true",
        help="required: acknowledgements that this overwrites a committed fixture",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"fixture directory to write (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    if not args.i_am_regenerating:
        print(
            "refusing to regenerate the committed graph fixture: pass "
            "--i-am-regenerating to acknowledge that it overwrites "
            f"{args.output} with a live extraction",
            file=sys.stderr,
        )
        return 2

    # The LLM preflight comes before any directory or database is created, so a
    # refused run cannot touch the committed fixture (tested).
    try:
        provider = get_llm_provider()
    except Exception as exc:  # noqa: BLE001 - the refusal is the point
        print(
            "refusing to regenerate the graph fixture: an LLM is required and "
            "none is available; configure a provider and its key "
            f"(DOCUGRAPH_LLM__PROVIDER, OPENAI_API_KEY, ...). Details: {exc}",
            file=sys.stderr,
        )
        return 2

    # Refuse an uncommitted extraction tree before any work: the manifest's
    # extraction_git_sha must identify the code that actually built the graph.
    dirty = _src_tree_dirty_reason()
    if dirty is not None:
        print(
            "refusing to regenerate the graph fixture: the manifest records the "
            "commit that built it, so src/ must be committed first. "
            f"{dirty}\nCommit or stash src/ and rerun.",
            file=sys.stderr,
        )
        return 2

    asyncio.run(_regenerate(args.output, provider))
    return 0


async def _regenerate(output: Path, provider: Any) -> None:
    from docugraph.storage.graph_store import GraphStore

    corpus = load_corpus(default_dataset_dir())
    work_dir = Path(tempfile.mkdtemp(prefix="docugraph-graph-fixture-"))
    store = GraphStore(work_dir / "graph")
    try:
        for doc in corpus:
            await store.add_episode(
                name=doc.id,
                content=doc.text,
                group_id=GRAPH_GROUP_ID,
                reference_time=INGEST_REFERENCE_TIME,
                source_description=f"bench corpus: {doc.title}",
            )
        output.mkdir(parents=True, exist_ok=True)
        await export_graph_fixture(store, output)
    finally:
        await store.close()
        shutil.rmtree(work_dir, ignore_errors=True)

    manifest = _manifest(output, provider)
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {output} ({len(corpus)} episodes)")


def _manifest(output: Path, provider: Any) -> dict[str, Any]:
    embedder = get_embedder()
    return {
        "fixture": FIXTURE_FORMAT,
        "group_id": GRAPH_GROUP_ID,
        "ingest_reference_time": INGEST_REFERENCE_TIME.isoformat().replace(
            "+00:00", "Z"
        ),
        "llm_provider": str(getattr(provider, "provider_name", "")),
        "llm_model": str(getattr(provider, "model_name", "")),
        "prompt_sha256": _prompt_sha256(),
        "graphiti_version": importlib_metadata.version("graphiti-core"),
        "extraction_git_sha": _git_sha(),
        "embedder": {
            "model": embedder.model_name,
            "dimensions": embedder.dimensions,
        },
        "corpus_sha256": {
            "corpus.jsonl": sha256_file(default_dataset_dir() / "corpus.jsonl"),
        },
        "sha256": {
            "graph.jsonl": sha256_file(output / "graph.jsonl"),
            "schema.cypher": sha256_file(output / "schema.cypher"),
        },
        "reproducibility_note": (
            "llm_model is a remotely served ':cloud' tag and may resolve to "
            "different weights or serving behaviour later. This committed "
            "fixture (identified by its sha256 map) is the reproducible "
            "artifact; regenerating it is not guaranteed to produce the same "
            "graph."
        ),
    }


def _prompt_sha256() -> str:
    """Hash Graphiti's prompt sources: the extraction contract, not its output."""
    import graphiti_core

    prompts = Path(graphiti_core.__file__).parent / "prompts"
    digest = hashlib.sha256()
    for path in sorted(prompts.rglob("*.py")):
        digest.update(path.relative_to(prompts).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _src_tree_dirty_reason() -> str | None:
    """Why ``src/`` is not a committed tree, or None when it is.

    Fails closed: if git cannot be asked, the builder refuses rather than
    guessing. Untracked files count -- they are code the commit does not name.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "src"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git could not be run to check src/: {exc}"
    if result.returncode != 0:
        return f"git status failed (exit {result.returncode}): {result.stderr.strip()}"
    dirty = result.stdout.strip()
    return f"src/ has uncommitted changes:\n{dirty}" if dirty else None


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
