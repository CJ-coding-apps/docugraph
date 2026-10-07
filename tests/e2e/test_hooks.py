"""The Claude Code hooks this repository ships, run for real.

`docugraph-auto-index.sh` is the only path by which a user's edited
documentation reaches the store without them typing a command, and it reports
success unconditionally -- the hook prints "Queued for automatic re-indexing"
whether or not anything was indexed. So the hook's own output cannot be the
evidence. The store is.

Nothing here is stubbed. The hook is executed as Claude Code executes it, with
a JSON document on stdin, and the assertion is made against the LanceDB the CLI
it spawns actually wrote to.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import lancedb

from docugraph.storage.vector_store import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTO_INDEX_HOOK = (
    REPO_ROOT / "claude-integration" / "hooks" / "docugraph-auto-index.sh"
)

# Comfortably over the chunker's 100-character `min_chunk_size`. That floor is
# not a detail to wave away: a shorter file is read, reported as "1 files", and
# yields zero chunks -- with the hook still reporting success. A fixture under
# it would exercise the chunker's floor rather than the hook's wiring.
DOC = """\
# The widget service

The widget service exposes an HTTP API for spinning widgets. It is deployed
behind the platform gateway, speaks JSON over HTTPS on port 8443, and requires
a bearer token issued by the identity service.

## Configuration

Set `WIDGET_TIMEOUT` to the number of seconds a spin may take. The default is
thirty seconds, which is long enough for the cold-start path on a small node
but short enough that a wedged spin is noticed before the client gives up.

## Failure modes

A spin that exceeds the timeout returns 504 and is recorded in the audit log
alongside the requesting tenant, so a noisy caller can be identified later.
"""


def _rows_in_store(data_dir: Path) -> int:
    """Rows in the CLI's LanceDB table, or 0 before the first write."""
    db = lancedb.connect(str(data_dir / "vectors"))
    if VectorStore.TABLE_NAME not in db.table_names():
        return 0
    return db.open_table(VectorStore.TABLE_NAME).count_rows()


def _wait_for_rows(data_dir: Path, timeout: float = 120.0) -> int:
    """Rows in the store, waiting for the hook's detached work to land.

    The hook backgrounds its indexing subshell and exits immediately, so the
    script returning does not mean the indexing finished. (`capture_output`
    below happens to hold the pipe open until that subshell closes it, so this
    usually returns on its first look -- but that is a property of how the
    test invokes the hook, not of the hook, and the wait is what actually
    depends on the indexing.)
    """
    deadline = time.monotonic() + timeout
    while True:
        rows = _rows_in_store(data_dir)
        if rows or time.monotonic() >= deadline:
            return rows
        time.sleep(0.5)


class TestTheAutoIndexHook:
    def test_editing_a_markdown_file_puts_chunks_in_the_store(self, tmp_path):
        """A doc file in, chunks in the store out.

        This is the test the hook's previous form needed and did not have. It
        passed the changed *file* to `docugraph index local`, whose argument is
        a directory (`click.Path(file_okay=False)`), along with a `--quiet` the
        command does not define. Click exited 2 on the unknown option and the
        `|| true` on the line swallowed it, so the hook announced that it had
        queued an indexing that never happened -- for every edit, silently.
        """
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "readme.md").write_text(DOC)

        # A temp data dir, never the real one. If the hook ignored the override
        # it would write to ~/.docugraph and this test would pass while proving
        # nothing about the path it names, which the existence check below
        # rules out.
        data_dir = tmp_path / "data"
        env = {
            **os.environ,
            "DOCUGRAPH_STORAGE__DATA_DIR": str(data_dir),
            # The hook's first act is `command -v docugraph`, so it needs the
            # console script on PATH. The interpreter running this test is the
            # one the package was installed into; its bin directory is where
            # that script lives.
            "PATH": os.pathsep.join(
                [str(Path(sys.executable).parent), os.environ.get("PATH", "")]
            ),
        }

        proc = subprocess.run(
            ["bash", str(AUTO_INDEX_HOOK)],
            input=json.dumps({"tool_input": {"file_path": str(docs / "readme.md")}}),
            text=True,
            capture_output=True,
            env=env,
            timeout=180,
        )

        # The hook prints its JSON reply and then, from the detached subshell,
        # whatever the CLI writes. Parse the reply off the front of that rather
        # than hoping it is the whole of stdout.
        reply, _ = json.JSONDecoder().raw_decode(proc.stdout)

        assert reply["continue"] is True, reply
        # Without this the test could pass on an early return -- a hook that did
        # not recognise the file as documentation still prints `continue: true`.
        assert "Queued for automatic re-indexing" in json.dumps(reply), reply

        rows = _wait_for_rows(data_dir)
        assert rows > 0, (
            f"the hook reported it had queued {docs / 'readme.md'}, and the store "
            f"it was pointed at holds {rows} rows.\n"
            f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
        )
        assert (data_dir / "vectors").is_dir(), (
            "the store is not where the hook was told to put it"
        )
