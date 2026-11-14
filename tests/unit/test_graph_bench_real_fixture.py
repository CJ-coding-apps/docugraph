"""Tests that need the real committed graph fixture (D1, D4, F1).

The staging rule is the point of this module. Until
``graph_fixture/MANIFEST.json`` is listed in ``benchmarks/retrieval/MANIFEST.json``
these tests SKIP with that reason; once it is listed, a missing, unreadable, or
invalid fixture FAILS. ``require_real_graph_fixture`` (in the shared helper)
implements exactly that rule, and it is called before anything imports the
not-yet-written ``docugraph.evals.graph_bench``, so the skip is observable
during the tests-first step.

No LLM key and no network: the determinism tests inject a deterministic
token-hash embedder that reports the fixture manifest's identity. Their numbers
are meaningless by construction -- D1 measures hash-seed determinism and D4
measures clock independence, not retrieval quality.
"""

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from docugraph.evals.dataset import load_corpus, load_queries, sha256_file

from .graph_bench_helpers import (
    DATASET,
    canonical_rankings,
    copy_fixture,
    fixture_identity_embedder,
    require_real_graph_fixture,
    run_fixture_arms,
    src_root,
)

# Determinism does not need every query; twelve distinct texts exercise the
# ranking, tie-breaks, and mapping without turning the test into a bench run.
DETERMINISM_QUERIES = 12


def _query_texts() -> list[str]:
    return [query.text for query in load_queries(DATASET)[:DETERMINISM_QUERIES]]


def _doc_ids() -> set[str]:
    return {doc.id for doc in load_corpus(DATASET)}


def _rank_real_fixture(fixture: Path, manifest: dict, queries: list[str], db_path: Path) -> dict:
    return run_fixture_arms(
        fixture,
        queries,
        db_path=db_path,
        embedder=fixture_identity_embedder(manifest),
        doc_ids=_doc_ids(),
    )


_CHILD_SCRIPT = """\
import asyncio
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, __SRC__)

from docugraph.core.embeddings import EmbeddingProviderBase
from docugraph.evals.graph_bench import (
    build_graph_arms,
    import_graph_fixture,
    load_fixture_manifest,
)
from docugraph.storage.graph_store import GraphStore


class HashedBag(EmbeddingProviderBase):
    def __init__(self, model_name, dimensions):
        self._model_name = model_name
        self._dimensions = dimensions

    def _one(self, text):
        vector = [0.0] * self._dimensions
        for token in text.casefold().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dimensions] += 1.0
        if not any(vector):
            vector[0] = 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed(self, texts):
        return [self._one(text) for text in texts]

    def embed_query(self, query):
        return self._one(query)

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def model_name(self):
        return self._model_name


def raising_llm():
    from graphiti_core.llm_client.client import LLMClient
    from graphiti_core.llm_client.config import LLMConfig

    class RaisingLLM(LLMClient):
        def __init__(self):
            super().__init__(config=LLMConfig(model="fake", small_model="fake"))

        async def _generate_response(self, *args, **kwargs):
            raise AssertionError("LLM used at query time")

    return RaisingLLM()


def main():
    fixture = Path(__FIXTURE__)
    manifest = load_fixture_manifest(fixture)
    embedder = HashedBag(
        manifest["embedder"]["model"], int(manifest["embedder"]["dimensions"])
    )
    queries = __QUERIES__
    doc_ids = set(__DOC_IDS__)
    sample = sorted(doc_ids)[:2] or ["stub-doc"]

    def _vector(query, top_k):
        return [(doc_id, 1.0) for doc_id in sample][:top_k]

    def _fts(query, top_k):
        return [(doc_id, 0.5) for doc_id in reversed(sample)][:top_k]

    loop = asyncio.new_event_loop()
    store = GraphStore(Path(__DB__), llm_client=raising_llm(), embedder=embedder)
    try:
        loop.run_until_complete(import_graph_fixture(store, fixture))
        arms = build_graph_arms(
            store=store,
            embedder=embedder,
            fixture_manifest=manifest,
            base_arms={"vector": _vector, "fts": _fts},
            doc_ids=doc_ids,
        )
        out = {}
        for name in sorted(arms):
            out[name] = [
                [
                    [chunk_id, round(float(score), 6)]
                    for chunk_id, score in arms[name](query, 10)
                ]
                for query in queries
            ]
    finally:
        loop.run_until_complete(store.close())
        loop.close()
    print(json.dumps(out, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


if __name__ == "__main__":
    main()
"""


class TestDeterminismAcrossProcesses:
    def test_three_hash_seeds_produce_byte_identical_rankings(self, tmp_path):
        """D1: three processes, three PYTHONHASHSEEDs, byte-identical output."""
        fixture = require_real_graph_fixture()
        script = _CHILD_SCRIPT
        script = script.replace("__SRC__", repr(str(src_root())))
        script = script.replace("__FIXTURE__", repr(str(fixture)))
        script = script.replace("__QUERIES__", repr(_query_texts()))
        script = script.replace("__DOC_IDS__", repr(sorted(_doc_ids())))
        script_path = tmp_path / "child.py"

        outputs = []
        for seed in ("0", "1", "2"):
            script_path.write_text(
                script.replace("__DB__", repr(str(tmp_path / f"db-{seed}"))),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONHASHSEED": seed,
                    "PYTHONPATH": str(src_root()),
                    "HOME": str(tmp_path / "home"),
                    "DOCUGRAPH_STORAGE__DATA_DIR": str(tmp_path / "data"),
                }
            )
            for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
                env.pop(key, None)
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(tmp_path),
                timeout=1200,
            )
            assert proc.returncode == 0, (
                f"child process with PYTHONHASHSEED={seed} failed:\n{proc.stderr}"
            )
            outputs.append(proc.stdout)

        assert outputs[0] == outputs[1], (
            "PYTHONHASHSEED=0 and PYTHONHASHSEED=1 produced different rankings"
        )
        assert outputs[0] == outputs[2], (
            "PYTHONHASHSEED=0 and PYTHONHASHSEED=2 produced different rankings"
        )


class TestClockIndependence:
    def test_output_is_identical_a_year_either_side(self, tmp_path):
        """D4: patch every graph-stack clock to 2025 and 2027; output equal."""
        fixture = require_real_graph_fixture()

        from docugraph.evals import graph_bench as gb

        manifest = gb.load_fixture_manifest(fixture)
        queries = _query_texts()

        baseline = canonical_rankings(
            _rank_real_fixture(fixture, manifest, queries, tmp_path / "db-base")
        )
        with _graph_clocks_at(2025):
            earlier = canonical_rankings(
                _rank_real_fixture(fixture, manifest, queries, tmp_path / "db-2025")
            )
        with _graph_clocks_at(2027):
            later = canonical_rankings(
                _rank_real_fixture(fixture, manifest, queries, tmp_path / "db-2027")
            )

        assert earlier == baseline, "clock at 2025 changed the ranked output"
        assert later == baseline, "clock at 2027 changed the ranked output"


@contextmanager
def _graph_clocks_at(year: int):
    """Point every clock binding reachable from the graph stack at one instant.

    The Phase 0 probe's method: graphiti 0.30.2 imports no clock in its search
    path, so this patches the names that do exist and restores them. A clock
    read introduced later would surface as a changed ranking, not a silent
    five-second drift.
    """
    import graphiti_core  # noqa: F401
    import graphiti_core.graphiti  # noqa: F401
    import graphiti_core.nodes  # noqa: F401
    import graphiti_core.utils.datetime_utils  # noqa: F401

    fixed = datetime(year, 1, 1, 0, 0, 0)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed if tz is None else fixed.replace(tzinfo=tz)

        @classmethod
        def utcnow(cls):
            return fixed

    saved: list[tuple[object, str, object]] = []
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if not name.startswith("graphiti_core"):
            continue
        if getattr(module, "utc_now", None) is not None:
            saved.append((module, "utc_now", module.utc_now))
            module.utc_now = lambda: fixed
        if getattr(module, "datetime", None) is datetime:
            saved.append((module, "datetime", module.datetime))
            module.datetime = FakeDatetime
    try:
        yield
    finally:
        for module, attribute, value in reversed(saved):
            setattr(module, attribute, value)


class TestManifestIntegrity:
    def test_one_byte_change_fails_the_manifest(self, tmp_path):
        """F1: alter the committed fixture by one byte; verification fails."""
        real = require_real_graph_fixture()
        tampered = copy_fixture(real, tmp_path / "graph_fixture")
        path = tampered / "graph.jsonl"
        data = bytearray(path.read_bytes())
        assert data, f"{path} is empty"
        data[0] ^= 0x01
        path.write_bytes(bytes(data))

        from docugraph.evals.graph_bench import verify_fixture_manifest

        problems = verify_fixture_manifest(tampered)
        assert any("graph.jsonl" in problem and "sha256" in problem for problem in problems), (
            f"tampering with graph.jsonl was not reported as a sha256 mismatch: {problems}"
        )

    def test_staged_fixture_manifest_verifies_clean(self):
        """Once staged, the committed fixture must pass its own check."""
        fixture = require_real_graph_fixture()

        from docugraph.evals.graph_bench import verify_fixture_manifest

        assert verify_fixture_manifest(fixture) == []

    def test_staged_fixture_hash_is_in_the_corpus_manifest(self):
        """The staging marker is itself an integrity-checked corpus file."""
        fixture = require_real_graph_fixture()
        corpus_manifest = json.loads((DATASET / "MANIFEST.json").read_text(encoding="utf-8"))
        expected = corpus_manifest["sha256"]["graph_fixture/MANIFEST.json"]
        actual = sha256_file(fixture / "MANIFEST.json")
        assert actual == expected, (
            f"graph_fixture/MANIFEST.json sha256 {actual} != corpus manifest {expected}"
        )
