"""Shared harness for the 0.5.0 graph-bench acceptance tests.

This module is not itself a contract: the test modules are. It exists so the
synthetic fixture, the deterministic embedders, the raising LLM client, and the
real-fixture staging rule are built the same way in every test file.

Two rules live here because the work order makes them binding:

* ``require_real_graph_fixture`` encodes the staging rule exactly. Until
  ``graph_fixture/MANIFEST.json`` appears in the corpus manifest the tests that
  need the real fixture are skipped with a reason; once it is listed, a missing
  or unreadable fixture is a *failure*, never a skip.
* ``build_synthetic_fixture`` produces a small P0-5-format fixture in
  ``tmp_path`` so the CI-bound tests never need the real graph, an embedding
  model, an LLM key, or the network.

Nothing here writes outside ``tmp_path`` (except the child process in the D1
test, which reads the real fixture and writes its own temporary store).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import shutil
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from docugraph.core.embeddings import EmbeddingProviderBase
from docugraph.evals.dataset import default_dataset_dir, sha256_file

DATASET = default_dataset_dir()
GRAPH_FIXTURE = DATASET / "graph_fixture"

# Fixed identity of the synthetic fixture's embedder. Any test that wants the
# graph arms to accept an embedder must report exactly these two values.
SYNTH_EMBEDDER_MODEL = "fake-bench-embedder"
SYNTH_DIM = 4
SYNTH_GROUP = "bench"
REFERENCE_TIME = "2026-01-01T00:00:00+00:00"
FIXTURE_FORMAT = "docugraph-graph-fixture/1"

# The corpus document ids the synthetic fixture stands for. The graph arms
# map an episode's ``name`` to a chunk id and must refuse a name outside this
# set (pre-registration, "Mapping to chunks/units of relevance").
SYNTH_DOC_IDS = frozenset(
    {"bench-doc-a", "bench-doc-b", "bench-doc-c", "bench-doc-d"}
)

# Every arm the work order defines. The fused arms are included: a test that
# says "every graph arm" means all five.
ALL_GRAPH_ARMS = frozenset(
    {
        "graph-entity",
        "graph-fact",
        "graph-2hop",
        "rrf+graph-2hop",
        "rrf+graph",
    }
)


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------


class KeywordEmbedder(EmbeddingProviderBase):
    """A deterministic 4-dimensional embedder with hand-known cosine values.

    ``alpha`` is axis 0, ``beta`` axis 1, ``gamma`` axis 2, ``delta`` axis 3.
    The synthetic document and entity embeddings are the same basis vectors,
    so a query for ``alpha`` has cosine 1.0 with entity ``Alpha`` and 0.0 with
    the rest -- rankings are readable by hand.
    """

    _AXIS = {"alpha": 0, "beta": 1, "gamma": 2, "delta": 3}

    def __init__(
        self,
        model_name: str = SYNTH_EMBEDDER_MODEL,
        dimensions: int = SYNTH_DIM,
    ) -> None:
        self._model_name = model_name
        self._dimensions = dimensions

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in text.casefold().split():
            axis = self._AXIS.get(token.strip(".,;:!?()[]{}\"'"))
            if axis is not None and axis < self._dimensions:
                vector[axis] += 1.0
        if not any(vector):
            vector[-1] = 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._one(query)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name


class HashedBagEmbedder(EmbeddingProviderBase):
    """Deterministic token-hash embeddings at an arbitrary dimension.

    Used only by the real-fixture determinism tests (D1, D4). It reports the
    fixture manifest's embedder identity so the graph arms accept it, but its
    vectors are meaningless by construction: those tests measure hash-seed and
    clock determinism, not retrieval quality, and this keeps them free of a
    model download.
    """

    def __init__(self, *, model_name: str, dimensions: int) -> None:
        self._model_name = model_name
        self._dimensions = dimensions

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in text.casefold().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dimensions] += 1.0
        if not any(vector):
            vector[0] = 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._one(query)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name


def fixture_identity_embedder(manifest: Mapping[str, Any]) -> HashedBagEmbedder:
    """A deterministic embedder reporting the manifest's identity.

    D1/D4 only. It is not a substitute for the frozen embedder in any test
    that makes a quality or identity claim.
    """
    embedder = manifest["embedder"]
    return HashedBagEmbedder(
        model_name=str(embedder["model"]),
        dimensions=int(embedder["dimensions"]),
    )


# ---------------------------------------------------------------------------
# The LLM client the query path must never touch
# ---------------------------------------------------------------------------


def make_raising_llm() -> Any:
    """A Graphiti LLMClient that counts and raises on any generation call.

    Subclassing is required: Graphiti validates injected clients with
    ``is_instance_of``. Construction touches nothing.
    """
    from graphiti_core.llm_client.client import LLMClient
    from graphiti_core.llm_client.config import LLMConfig

    class RaisingLLM(LLMClient):
        calls = 0

        def __init__(self) -> None:
            super().__init__(config=LLMConfig(model="fake", small_model="fake"))

        async def _generate_response(
            self,
            messages: Any,  # noqa: ARG002
            response_model: Any = None,  # noqa: ARG002
            max_tokens: Any = None,  # noqa: ARG002
            model_size: Any = None,  # noqa: ARG002
        ) -> Any:
            type(self).calls += 1
            raise AssertionError("LLM client used at query time (_generate_response)")

        async def generate_response(self, *args: Any, **kwargs: Any) -> Any:
            type(self).calls += 1
            raise AssertionError("LLM client used at query time (generate_response)")

    return RaisingLLM()


# ---------------------------------------------------------------------------
# The synthetic P0-5 fixture
# ---------------------------------------------------------------------------


def _node(table: str, row: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "node", "table": table, "row": row}


def _rel(
    table: str,
    src_label: str,
    src: str,
    dst_label: str,
    dst: str,
    row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "rel",
        "table": table,
        "src_label": src_label,
        "src": src,
        "dst_label": dst_label,
        "dst": dst,
        "row": row or {},
    }


def _entity(
    uuid: str,
    name: str,
    group_id: str,
    embedding: list[float],
) -> dict[str, Any]:
    return _node(
        "Entity",
        {
            "uuid": uuid,
            "name": name,
            "group_id": group_id,
            "labels": ["Entity"],
            "created_at": REFERENCE_TIME,
            "name_embedding": embedding,
            "summary": "",
            "attributes": "{}",
        },
    )


def _episode(uuid: str, name: str, group_id: str, content: str) -> dict[str, Any]:
    return _node(
        "Episodic",
        {
            "uuid": uuid,
            "name": name,
            "group_id": group_id,
            "created_at": REFERENCE_TIME,
            "source": "text",
            "source_description": content,
            "content": content,
            "valid_at": REFERENCE_TIME,
            "entity_edges": [],
        },
    )


def _fact(
    uuid: str,
    fact: str,
    embedding: list[float],
    episode_uuid: str,
    *,
    invalid_at: str | None = None,
    expired_at: str | None = None,
) -> dict[str, Any]:
    return _node(
        "RelatesToNode_",
        {
            "uuid": uuid,
            "group_id": SYNTH_GROUP,
            "created_at": REFERENCE_TIME,
            "name": fact,
            "fact": fact,
            "fact_embedding": embedding,
            "episodes": [episode_uuid],
            "expired_at": expired_at,
            "valid_at": REFERENCE_TIME,
            "invalid_at": invalid_at,
            "reference_time": REFERENCE_TIME,
            "attributes": "{}",
        },
    )


def _mention(uuid: str, episode_uuid: str, entity_uuid: str, group_id: str) -> dict[str, Any]:
    return _rel(
        "MENTIONS",
        "Episodic",
        episode_uuid,
        "Entity",
        entity_uuid,
        {"uuid": uuid, "group_id": group_id, "created_at": REFERENCE_TIME},
    )


def _relates(entity_uuid: str, fact_uuid: str) -> dict[str, Any]:
    return _rel("RELATES_TO", "Entity", entity_uuid, "RelatesToNode_", fact_uuid)


def _relates_reverse(fact_uuid: str, entity_uuid: str) -> dict[str, Any]:
    return _rel("RELATES_TO", "RelatesToNode_", fact_uuid, "Entity", entity_uuid)


def synthetic_records(
    *,
    extra_nodes: Sequence[dict[str, Any]] = (),
    extra_rels: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """A four-document, four-entity, two-fact graph, all in group ``bench``.

    Plus one decoy episode and entity in group ``other`` with an embedding
    identical to the query for ``alpha``: N1 uses it to prove every graph arm
    scopes its search to ``group_ids=["bench"]``.
    """
    nodes = [
        _entity("ent-alpha", "Alpha", SYNTH_GROUP, [1.0, 0.0, 0.0, 0.0]),
        _entity("ent-beta", "Beta", SYNTH_GROUP, [0.0, 1.0, 0.0, 0.0]),
        _entity("ent-gamma", "Gamma", SYNTH_GROUP, [0.0, 0.0, 1.0, 0.0]),
        _entity("ent-delta", "Delta", SYNTH_GROUP, [0.0, 0.0, 0.0, 1.0]),
        _entity("ent-decoy", "Alpha Decoy", "other", [1.0, 0.0, 0.0, 0.0]),
        _episode("ep-doc-a", "bench-doc-a", SYNTH_GROUP, "Alpha document"),
        _episode("ep-doc-b", "bench-doc-b", SYNTH_GROUP, "Beta document"),
        _episode("ep-doc-c", "bench-doc-c", SYNTH_GROUP, "Gamma document"),
        _episode("ep-doc-d", "bench-doc-d", SYNTH_GROUP, "Delta document"),
        _episode("ep-decoy", "decoy-doc", "other", "Decoy document"),
        _fact(
            "fact-alpha-beta",
            "Alpha links to Beta",
            [0.5, 0.5, 0.0, 0.0],
            "ep-doc-a",
        ),
        _fact(
            "fact-gamma-delta",
            "Gamma links to Delta",
            [0.0, 0.0, 0.5, 0.5],
            "ep-doc-c",
            invalid_at="2026-06-01T00:00:00+00:00",
            expired_at="2026-06-02T00:00:00+00:00",
        ),
        *extra_nodes,
    ]
    rels = [
        _mention("m-a-alpha", "ep-doc-a", "ent-alpha", SYNTH_GROUP),
        _mention("m-b-beta", "ep-doc-b", "ent-beta", SYNTH_GROUP),
        _mention("m-c-gamma", "ep-doc-c", "ent-gamma", SYNTH_GROUP),
        _mention("m-d-delta", "ep-doc-d", "ent-delta", SYNTH_GROUP),
        _mention("m-decoy", "ep-decoy", "ent-decoy", "other"),
        _relates("ent-alpha", "fact-alpha-beta"),
        _relates_reverse("fact-alpha-beta", "ent-beta"),
        _relates("ent-gamma", "fact-gamma-delta"),
        _relates_reverse("fact-gamma-delta", "ent-delta"),
        *extra_rels,
    ]
    return nodes + rels


def installed_graphiti_version() -> str:
    from importlib import metadata as im

    return im.version("graphiti-core")


def synthetic_manifest(fixture_dir: Path, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "fixture": FIXTURE_FORMAT,
        "group_id": SYNTH_GROUP,
        "ingest_reference_time": REFERENCE_TIME,
        "llm_model": "fake-extractor",
        "graphiti_version": installed_graphiti_version(),
        "prompt_sha256": "0" * 64,
        "extraction_git_sha": "0" * 40,
        "embedder": {"model": SYNTH_EMBEDDER_MODEL, "dimensions": SYNTH_DIM},
        "sha256": {
            "graph.jsonl": sha256_file(fixture_dir / "graph.jsonl"),
            "schema.cypher": sha256_file(fixture_dir / "schema.cypher"),
        },
    }


def build_synthetic_fixture(
    root: Path,
    *,
    records: Sequence[dict[str, Any]] | None = None,
) -> Path:
    """Write a P0-5 fixture under ``root/graph_fixture`` and return its path.

    ``schema.cypher`` comes from ``export_graph_fixture`` on an empty, freshly
    initialised store, so the DDL is exactly what the installed graphiti emits;
    hand-writing it would pin a schema the importer does not share. The data
    rows are the hand-written records above.

    Requires the Q1 ``llm_client=`` / ``embedder=`` construction arguments on
    ``GraphStore`` (no configured LLM, no model download).
    """
    from docugraph.evals.graph_bench import export_graph_fixture
    from docugraph.storage.graph_store import GraphStore

    fixture_dir = root / "graph_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    rows = list(records) if records is not None else synthetic_records()

    loop = asyncio.new_event_loop()
    try:
        store = GraphStore(
            root / "schema_source",
            llm_client=make_raising_llm(),
            embedder=KeywordEmbedder(),
        )
        try:
            loop.run_until_complete(export_graph_fixture(store, fixture_dir))
        finally:
            loop.run_until_complete(store.close())
    finally:
        loop.close()

    (fixture_dir / "graph.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in rows),
        encoding="utf-8",
    )
    manifest = synthetic_manifest(fixture_dir, rows)
    (fixture_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return fixture_dir


# ---------------------------------------------------------------------------
# Running the arms over a fixture (sync test code, async setup)
# ---------------------------------------------------------------------------


def stub_base_arms(doc_ids: Collection[str]) -> dict[str, Any]:
    """The two legs a fused graph arm needs, without a LanceDB store.

    The fused arms' job in these tests is to prove they run and stay
    deterministic; their base legs may be anything stable. The ids come from
    the corpus so a fused arm never sees an id its own graph cannot address.
    """
    from docugraph.evals.arms import Arm

    sample = sorted(doc_ids)[:2] or ["stub-doc"]

    def vector(query: str, top_k: int) -> list[tuple[str, float]]:
        return [(doc_id, 1.0) for doc_id in sample][:top_k]

    def fts(query: str, top_k: int) -> list[tuple[str, float]]:
        return [(doc_id, 0.5) for doc_id in reversed(sample)][:top_k]

    arms: dict[str, Arm] = {"vector": vector, "fts": fts}
    return arms


def run_fixture_arms(
    fixture_dir: Path,
    queries: Sequence[str],
    *,
    db_path: Path,
    embedder: EmbeddingProviderBase,
    doc_ids: Collection[str],
    llm_client: Any | None = None,
    base_arms: Mapping[str, Any] | None = None,
    names: Sequence[str] | None = None,
    top_k: int = 10,
) -> dict[str, list[list[tuple[str, float]]]]:
    """Import ``fixture_dir`` into a fresh store and run the graph arms.

    Setup is async; the arms are the sync ``Arm`` callables the existing runner
    expects, so they are invoked with no running event loop.
    """
    from docugraph.evals.graph_bench import (
        build_graph_arms,
        import_graph_fixture,
        load_fixture_manifest,
    )
    from docugraph.storage.graph_store import GraphStore

    loop = asyncio.new_event_loop()
    store = GraphStore(
        db_path,
        llm_client=llm_client if llm_client is not None else make_raising_llm(),
        embedder=embedder,
    )
    try:
        loop.run_until_complete(import_graph_fixture(store, fixture_dir))
        arms = build_graph_arms(
            store=store,
            embedder=embedder,
            fixture_manifest=load_fixture_manifest(fixture_dir),
            base_arms=base_arms if base_arms is not None else stub_base_arms(doc_ids),
            doc_ids=doc_ids,
        )
        if names is not None:
            arms = {name: arms[name] for name in names}
        return {
            name: [arm(query, top_k) for query in queries]
            for name, arm in arms.items()
        }
    finally:
        loop.run_until_complete(store.close())
        loop.close()


def canonical_rankings(
    rankings: Mapping[str, Sequence[Sequence[tuple[str, float]]]],
) -> str:
    """Ranked ids and scores rounded to 1e-6, in canonical JSON (P0-4 form)."""
    payload = {
        name: [
            [[chunk_id, round(float(score), 6)] for chunk_id, score in row]
            for row in rows
        ]
        for name, rows in rankings.items()
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------------
# Staging rule
# ---------------------------------------------------------------------------


def require_real_graph_fixture() -> Path:
    """The committed graph fixture, or a skip until it is staged.

    Exactly the rule the work order sets out: while
    ``graph_fixture/MANIFEST.json`` is absent from the corpus manifest the
    real-fixture tests skip with a reason; once it is listed, a missing,
    unreadable, or invalid fixture is a failure.
    """
    corpus_manifest_path = DATASET / "MANIFEST.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    listed = "graph_fixture/MANIFEST.json" in (corpus_manifest.get("sha256") or {})
    if not listed:
        pytest.skip(
            "the graph fixture is not staged yet: add 'graph_fixture/MANIFEST.json' "
            f"to {corpus_manifest_path} when the fixture is regenerated"
        )

    from docugraph.evals.graph_bench import verify_fixture_manifest

    manifest_path = GRAPH_FIXTURE / "MANIFEST.json"
    if not manifest_path.is_file():
        pytest.fail(
            f"{manifest_path} is listed in {corpus_manifest_path} but does not exist"
        )
    try:
        problems = verify_fixture_manifest(GRAPH_FIXTURE)
    except Exception as exc:  # noqa: BLE001 - any read failure is a hard failure
        pytest.fail(f"graph fixture at {GRAPH_FIXTURE} is unreadable: {exc}")
    if problems:
        pytest.fail(
            "graph fixture manifest does not verify:\n"
            + "\n".join(f"- {problem}" for problem in problems)
        )
    return GRAPH_FIXTURE


def copy_fixture(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def normalise_export_value(value: Any) -> Any:
    """Make an exported value comparable to its hand-written source.

    Timestamps: Kùzu stores naive TIMESTAMPs, so an ISO-8601 input with an
    offset comes back without one. Compare instants, not spellings. Floats and
    lists are compared exactly -- the fixture's embeddings are dyadic rationals,
    exactly representable in float32, so a re-embed would show up here.
    """
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return value
        return parsed.replace(tzinfo=None)
    if isinstance(value, list):
        return [normalise_export_value(item) for item in value]
    return value


def record_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    if record["kind"] == "node":
        return ("node", str(record["table"]), str(record["row"]["uuid"]))
    return (
        "rel",
        str(record["table"]),
        str(record["src"]),
        str(record["dst"]),
    )


def src_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src"
