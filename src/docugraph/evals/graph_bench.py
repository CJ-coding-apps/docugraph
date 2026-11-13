"""Graph-augmented retrieval arms and the pre-registered gate (WP §4-§5).

Everything the pre-registration fixes is a module constant here, so a test can
assert the registered literal and a reader can see the number the numbers were
judged against. Nothing in this module is tuned on results: the arms follow the
published recipe, the gate follows the published outcome mapping, and the
fidelity row is declared non-gating in code as well as in prose.

Two properties are structural rather than tested-into-existence:

* **No LLM at query time.** The arms read entity/fact tables and the frozen
  embedder; ``graph-fact`` uses Graphiti's edge RRF recipe, whose reranker is
  RRF and whose cross-encoder is the store's ``NoopCrossEncoder``. A raising
  LLM client is attached by the tests and must never be called.
* **The committed fixture is the only input.** Import copies rows from the
  fixture text; nothing is re-embedded, re-extracted, or recomputed. A store
  opened for the bench may therefore have no reachable LLM at all.

The read helpers on ``GraphStore`` (``_search_nodes``, ``_list_episodes``,
``_edges_for_nodes``) are marked "eval use; not a stable API": they exist so
the arms read the graph in one place, not so other code can depend on them.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import shutil
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from docugraph.evals.dataset import sha256_file

# --- Registered numbers (docs/graph-bench-preregistration-2026-11-13.md) ---

GRAPH_GROUP_ID = "bench"
SEED_CAP = 20
EDGE_CAP = 50
RRF_K = 60

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261113

NDCG_THRESHOLD = 0.05
SET_RECALL_THRESHOLD = 0.10
REGRESSION_FLOOR = -0.02
CATEGORY_FLAG_FLOOR = -0.05

GATE_ARM = "rrf+graph-2hop"
FALLBACK_GATE_ARM = "rrf+graph"

OUTCOME_MET = "MET"
OUTCOME_NOT_MET = "NOT MET"
OUTCOME_INCONCLUSIVE = "INCONCLUSIVE"

GRAPH_ARM_NAMES = (
    "graph-entity",
    "graph-fact",
    "graph-2hop",
    "rrf+graph-2hop",
    "rrf+graph",
)

FIXTURE_FORMAT = "docugraph-graph-fixture/1"

#: Fields the manifest must carry for the fixture to be reproducible at all.
_REQUIRED_MANIFEST_FIELDS = (
    "fixture",
    "group_id",
    "ingest_reference_time",
    "llm_model",
    "graphiti_version",
    "prompt_sha256",
    "extraction_git_sha",
    "embedder",
)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class GraphFixtureError(ValueError):
    """The committed fixture is missing, unreadable, or does not verify."""


class GraphArmUnavailable(RuntimeError):
    """A graph arm refused to run (embedder mismatch, per R-1)."""


@dataclass(frozen=True)
class GraphArmStatus:
    """Whether one arm may run, and why not when it may not.

    A status, not an exception, for reporting: every arm is listed with its
    reason, so a report can say which arms ran and which refused, instead of a
    single traceback that hides the others.
    """

    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class MetricResult:
    """A paired mean delta with its percentile bootstrap interval."""

    point: float
    ci_lower: float
    ci_upper: float
    threshold: float


# ---------------------------------------------------------------------------
# Manifest and fixture IO
# ---------------------------------------------------------------------------


def load_fixture_manifest(fixture_dir: Path | str) -> dict[str, Any]:
    """Read ``MANIFEST.json`` without judging it (``verify`` does that)."""
    path = Path(fixture_dir) / "MANIFEST.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise GraphFixtureError(f"{path} does not hold a JSON object")
        return manifest
    except FileNotFoundError as exc:
        raise GraphFixtureError(f"{path} is missing") from exc
    except json.JSONDecodeError as exc:
        raise GraphFixtureError(f"{path} is not valid JSON: {exc}") from exc


def verify_fixture_manifest(fixture_dir: Path | str) -> list[str]:
    """Every way the fixture's manifest can be wrong, in one list.

    Content is checked by hash **before** it is parsed: F1 flips the first byte
    of ``graph.jsonl``, which makes it invalid JSON as well as a hash mismatch,
    and the report must name the hash -- the integrity claim -- not a parser
    error. An empty list means the fixture verifies.
    """
    fixture_dir = Path(fixture_dir)
    manifest_path = fixture_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        return [f"{manifest_path} is missing"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{manifest_path} is not valid JSON: {exc}"]

    problems: list[str] = []
    if manifest.get("fixture") != FIXTURE_FORMAT:
        problems.append(
            f"{manifest_path}: fixture format {manifest.get('fixture')!r} "
            f"!= {FIXTURE_FORMAT!r}"
        )
    for field in _REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            problems.append(f"{manifest_path}: field {field!r} is missing")

    if manifest.get("group_id") != GRAPH_GROUP_ID:
        problems.append(
            f"{manifest_path}: group_id {manifest.get('group_id')!r} "
            f"!= {GRAPH_GROUP_ID!r}"
        )

    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        problems.append(f"{manifest_path}: no sha256 map")
    else:
        for relative, expected in sorted(hashes.items()):
            path = fixture_dir / relative
            if not path.is_file():
                problems.append(f"{path} is missing (listed in {manifest_path})")
                continue
            actual = sha256_file(path)
            if actual != expected:
                problems.append(
                    f"{path}: sha256 {actual} != manifest {expected}"
                )

    installed = _installed_graphiti_version()
    declared = manifest.get("graphiti_version")
    if declared != installed:
        problems.append(
            f"{manifest_path}: graphiti_version {declared!r} "
            f"does not match the installed {installed!r}"
        )

    embedder = manifest.get("embedder")
    if not isinstance(embedder, dict) or not embedder.get("model"):
        problems.append(f"{manifest_path}: embedder.model is missing")

    return problems


def _installed_graphiti_version() -> str:
    try:
        return importlib_metadata.version("graphiti-core")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover - hard dep
        return "not-installed"


def load_graph_fixture(fixture_dir: Path | str) -> list[dict[str, Any]]:
    """Parse ``graph.jsonl`` into records (nodes first, then rels)."""
    path = Path(fixture_dir) / "graph.jsonl"
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise GraphFixtureError(f"{path}:{line_no}: {exc}") from exc
    return records


async def import_graph_fixture(store: Any, fixture_dir: Path | str) -> None:
    """Recreate the fixture's graph in ``store``'s database, through its driver.

    Integrity is a separate, explicit step: the bench calls
    ``verify_fixture_manifest`` before importing (the staging rule does exactly
    that), and an export that is being re-imported in the round-trip test is a
    valid input even though a manifest copied to it describes the *source*
    bytes. Import is a data operation, not an integrity gate; F1 exercises the
    gate directly.

    The store's own initialization has already created Graphiti's schema and
    FTS indexes; ``schema.cypher`` is the committed, diffable record of that
    DDL, not a second thing to execute (executing it would collide with the
    tables initialization just made).
    """
    fixture_dir = Path(fixture_dir)
    records = load_graph_fixture(fixture_dir)

    node_records = [record for record in records if record["kind"] == "node"]
    rel_records = [record for record in records if record["kind"] == "rel"]

    driver = await _driver_for(store)

    for record in sorted(node_records, key=_record_key):
        table = _checked_identifier(record["table"])
        row = {key: _from_json(value) for key, value in record["row"].items()}
        columns = ", ".join(f"`{key}`: ${key}" for key in row)
        await driver.execute_query(
            f"CREATE (:`{table}` {{{columns}}})",
            **row,
        )

    for record in sorted(rel_records, key=_record_key):
        table = _checked_identifier(record["table"])
        src_label = _checked_identifier(record["src_label"])
        dst_label = _checked_identifier(record["dst_label"])
        row = {key: _from_json(value) for key, value in record["row"].items()}
        params = dict(row, src=record["src"], dst=record["dst"])
        properties = ", ".join(f"`{key}`: ${key}" for key in row)
        suffix = f" {{{properties}}}" if properties else ""
        await driver.execute_query(
            f"MATCH (a:`{src_label}` {{uuid: $src}}), "
            f"(b:`{dst_label}` {{uuid: $dst}}) "
            f"CREATE (a)-[:`{table}`{suffix}]->(b)",
            **params,
        )


async def export_graph_fixture(store: Any, out_dir: Path | str) -> None:
    """Write the fixture format: ``schema.cypher`` + ``graph.jsonl``.

    ``schema.cypher`` comes from Kuzu's own ``EXPORT DATABASE`` (the DDL the
    importer's schema was created from, not a hand copy). The data is JSONL
    because Kuzu's CSV writer rounds floats to six decimals, which silently
    corrupts embeddings; ``json`` repr round-trips a float32 exactly.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    driver = await _driver_for(store)

    ddl_dir = out_dir / "_ddl"
    shutil.rmtree(ddl_dir, ignore_errors=True)
    escaped = str(ddl_dir).replace("'", "\\'")
    await driver.execute_query(f"EXPORT DATABASE '{escaped}' (format='csv')")
    (out_dir / "schema.cypher").write_text(
        (ddl_dir / "schema.cypher").read_text(encoding="utf-8"), encoding="utf-8"
    )
    shutil.rmtree(ddl_dir, ignore_errors=True)

    tables, _, _ = await driver.execute_query("CALL show_tables() RETURN name, type;")
    kinds = {row["name"]: row["type"] for row in tables or []}

    records: list[dict[str, Any]] = []
    for table in sorted(kinds):
        if kinds[table] == "NODE":
            rows, _, _ = await driver.execute_query(
                f"MATCH (n:`{table}`) RETURN n.*;"
            )
            for row in rows or []:
                records.append(
                    {
                        "kind": "node",
                        "table": table,
                        "row": {
                            key.split(".", 1)[-1]: _jsonable(value)
                            for key, value in row.items()
                        },
                    }
                )
        else:
            rows, _, _ = await driver.execute_query(
                f"MATCH (a)-[r:`{table}`]->(b) "
                "RETURN label(a) AS src_label, label(b) AS dst_label, "
                "a.uuid AS src, b.uuid AS dst, r.*;"
            )
            for row in rows or []:
                row = dict(row)
                src = row.pop("src")
                dst = row.pop("dst")
                src_label = row.pop("src_label")
                dst_label = row.pop("dst_label")
                records.append(
                    {
                        "kind": "rel",
                        "table": table,
                        "src": src,
                        "dst": dst,
                        "src_label": src_label,
                        "dst_label": dst_label,
                        "row": {
                            key.split(".", 1)[-1]: _jsonable(value)
                            for key, value in row.items()
                        },
                    }
                )

    records.sort(key=_record_key)
    with (out_dir / "graph.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")


async def _driver_for(store: Any) -> Any:
    """Initialize the store and hand back its driver.

    Import/export are statements about the *database*, not about Graphiti's
    object model, so they go through the driver directly. Initialization is
    awaited here so the schema exists before the first row.
    """
    await store._ensure_initialized()
    driver = getattr(store, "_driver", None)
    if driver is None:  # pragma: no cover - initialization always sets it
        raise GraphFixtureError("the store has no driver after initialization")
    return driver


def _record_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    if record["kind"] == "node":
        return ("node", str(record["table"]), str(record["row"]["uuid"]))
    return ("rel", str(record["table"]), str(record["src"]), str(record["dst"]))


def _checked_identifier(name: Any) -> str:
    value = str(name)
    if not _IDENTIFIER.match(value):
        raise GraphFixtureError(f"unsafe table or label name in fixture: {value!r}")
    return value


def _jsonable(value: Any) -> Any:
    """Kuzu value -> JSON value (timestamps as ISO-8601, NaN as ``"NaN"``)."""
    if value is None:
        return None
    if isinstance(value, float):
        return "NaN" if math.isnan(value) else value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, int, str)):
        return value
    # numpy scalars and similar: convert through the closest Python type.
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _from_json(value: Any) -> Any:
    """JSON value -> Kuzu parameter (the inverse of ``_jsonable``)."""
    if value == "NaN":
        return float("nan")
    if isinstance(value, str) and len(value) >= 10 and value[4] == "-" and value[7] == "-":
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value
    if isinstance(value, list):
        return [_from_json(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def _embedder_refusal(
    fixture_manifest: Mapping[str, Any], embedder: Any
) -> str | None:
    """Why this embedder cannot stand in for the fixture's, or ``None``.

    Both the model id and the dimension must match: the same width with a
    different id is a different vector space (R-4), and a cloud embedder that
    happens to be 1536-dimensional is not the frozen one.
    """
    declared = fixture_manifest.get("embedder") or {}
    fixture_model = str(declared.get("model", ""))
    fixture_dimensions = declared.get("dimensions")
    model = str(getattr(embedder, "model_name", ""))
    dimensions = getattr(embedder, "dimensions", None)

    if model == fixture_model and dimensions == fixture_dimensions:
        return None
    return (
        f"the fixture was built with embedder {fixture_model!r} "
        f"({fixture_dimensions} dimensions), but the selected embedder is "
        f"{model!r} ({dimensions} dimensions); graph arms refuse to mix "
        "vector spaces"
    )


def graph_arm_statuses(
    *,
    fixture_manifest: Mapping[str, Any],
    embedder: Any,
) -> dict[str, GraphArmStatus]:
    """Every graph arm with its availability, so a report can list them all."""
    refusal = _embedder_refusal(fixture_manifest, embedder)
    return {
        name: GraphArmStatus(available=refusal is None, reason=refusal)
        for name in GRAPH_ARM_NAMES
    }


def build_graph_arms(
    *,
    store: Any,
    embedder: Any,
    fixture_manifest: Mapping[str, Any],
    base_arms: Mapping[str, Any],
    doc_ids: Collection[str],
) -> dict[str, Any]:
    """The five graph arms as sync ``Arm`` callables, or a refusal.

    The embedder is checked before the store is touched: a mismatch is a
    property of the request, and the test that proves it passes ``store=None``.
    """
    statuses = graph_arm_statuses(fixture_manifest=fixture_manifest, embedder=embedder)
    refused = [status for status in statuses.values() if not status.available]
    if refused:
        raise GraphArmUnavailable(refused[0].reason or "graph arms unavailable")

    doc_id_set = frozenset(str(doc_id) for doc_id in doc_ids)
    cache: list[_GraphSnapshot] = []

    def get_snapshot() -> _GraphSnapshot:
        if not cache:
            cache.append(asyncio.run(_load_snapshot(store, doc_id_set)))
        return cache[0]

    def graph_entity(query: str, top_k: int) -> list[tuple[str, float]]:
        return _expand(get_snapshot(), embedder, query, hops=1, top_k=top_k)

    def graph_2hop(query: str, top_k: int) -> list[tuple[str, float]]:
        return _expand(get_snapshot(), embedder, query, hops=2, top_k=top_k)

    def graph_fact(query: str, top_k: int) -> list[tuple[str, float]]:
        return asyncio.run(_fact_arm(store, get_snapshot(), query, top_k))

    def rrf_plus_graph_2hop(query: str, top_k: int) -> list[tuple[str, float]]:
        return _fuse(
            _base_rrf(base_arms, query, top_k),
            graph_2hop(query, top_k),
            top_k=top_k,
        )

    def rrf_plus_graph(query: str, top_k: int) -> list[tuple[str, float]]:
        return _fuse(
            _base_rrf(base_arms, query, top_k),
            graph_entity(query, top_k),
            top_k=top_k,
        )

    return {
        "graph-entity": graph_entity,
        "graph-fact": graph_fact,
        "graph-2hop": graph_2hop,
        "rrf+graph-2hop": rrf_plus_graph_2hop,
        "rrf+graph": rrf_plus_graph,
    }


@dataclass(frozen=True)
class _GraphSnapshot:
    """The bench namespace, read once: entities, facts, episode names."""

    entities: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    episode_names: dict[str, str]
    mentions: dict[str, list[dict[str, str]]]


async def _load_snapshot(store: Any, doc_ids: frozenset[str]) -> _GraphSnapshot:
    """Read the bench graph and prove every episode maps to a corpus id.

    A name that does not resolve is an error, not a dropped result: silently
    dropping it would let an arm score against a chunk space it cannot address.
    The check runs once, at snapshot time, so a broken mapping fails before a
    single query is scored.
    """
    episodes = await store._list_episodes([GRAPH_GROUP_ID])
    episode_names = {row["uuid"]: row["name"] for row in episodes}
    for row in episodes:
        if row["name"] not in doc_ids:
            raise ValueError(
                f"episode {row['name']!r} does not map to a corpus document id; "
                "the fixture and the corpus disagree"
            )

    entities = await store._search_nodes([GRAPH_GROUP_ID])
    mentions = {
        entity["uuid"]: entity.pop("episodes", []) for entity in entities
    }
    facts = await store._edges_for_nodes(
        [entity["uuid"] for entity in entities], [GRAPH_GROUP_ID]
    )
    return _GraphSnapshot(
        entities=entities,
        facts=facts,
        episode_names=episode_names,
        mentions=mentions,
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def _expand(
    snapshot: _GraphSnapshot,
    embedder: Any,
    query: str,
    *,
    hops: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """Entity-seeded expansion with the pre-registered fan-out and ranking.

    Seeds are the top ``SEED_CAP`` entities by ``(-cosine, uuid)``. Chunks at
    distance 1 are episodes mentioning a seed; at distance 2 they are episodes
    mentioning the far end of a kept fact, excluding that fact's own episode
    (it belongs to the seed's document). A chunk keeps its best offering by
    ``(distance, -seed score, -edge similarity, chunk_id)``.

    Scores returned to the caller are rank-derived: the registered object is an
    *ordering*, and the metrics consume order. Deriving the score from the rank
    makes the ordering survive ``stable_ranking`` unchanged, instead of letting
    a tie-break on a similarity number reorder the registered ranking.
    """
    query_vector = embedder.embed_query(query)

    scored_entities = [
        (entity["uuid"], _cosine(query_vector, entity["name_embedding"]))
        for entity in snapshot.entities
    ]
    scored_entities.sort(key=lambda pair: (-pair[1], pair[0]))
    seeds = scored_entities[:SEED_CAP]
    seed_score = dict(seeds)

    chunk: dict[str, tuple[int, float, float]] = {}

    def offer(name: str, candidate: tuple[int, float, float]) -> None:
        current = chunk.get(name)
        key = (candidate[0], -candidate[1], -candidate[2])
        if current is None or key < (current[0], -current[1], -current[2]):
            chunk[name] = candidate

    for uuid, score in seeds:
        for episode in snapshot.mentions.get(uuid, ()):
            offer(episode["name"], (1, score, 0.0))

    if hops >= 2:
        candidates = [
            (fact["uuid"], fact["source_uuid"], _cosine(query_vector, fact["fact_embedding"]))
            for fact in snapshot.facts
            if fact["source_uuid"] in seed_score
        ]
        candidates.sort(key=lambda triple: (-triple[2], triple[0]))
        kept = candidates[:EDGE_CAP]

        best_by_target: dict[str, tuple[float, float, frozenset[str]]] = {}
        for fact_uuid, source_uuid, similarity in kept:
            fact = next(item for item in snapshot.facts if item["uuid"] == fact_uuid)
            target = fact["target_uuid"]
            pair = (seed_score[source_uuid], similarity, frozenset(fact["episodes"]))
            current = best_by_target.get(target)
            if current is None or (-pair[0], -pair[1]) < (-current[0], -current[1]):
                best_by_target[target] = pair

        for target, (seed, similarity, own_episodes) in best_by_target.items():
            for episode in snapshot.mentions.get(target, ()):
                if episode["uuid"] in own_episodes:
                    continue
                offer(episode["name"], (2, seed, similarity))

    ranked = sorted(chunk.items(), key=lambda item: (item[1][0], -item[1][1], -item[1][2], item[0]))
    ordered = ranked[:top_k]
    return [
        (name, float(len(ordered) - index))
        for index, (name, _) in enumerate(ordered)
    ]


async def _fact_arm(
    store: Any,
    snapshot: _GraphSnapshot,
    query: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """``graph-fact``: Graphiti's edge RRF recipe, mapped through episodes.

    Diagnostic only. Its reranker is RRF and the store's cross-encoder is the
    no-op, so no LLM is reachable from this path; the raising client in test D2
    proves it.
    """
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

    graphiti = await store._ensure_initialized()
    config = EDGE_HYBRID_SEARCH_RRF.model_copy(update={"limit": top_k})
    result = await graphiti.search_(
        query, config=config, group_ids=[GRAPH_GROUP_ID]
    )

    ordered: list[str] = []
    seen: set[str] = set()
    for edge in result.edges:
        for episode_uuid in edge.episodes:
            name = snapshot.episode_names.get(episode_uuid)
            if name is None or name in seen:
                continue
            seen.add(name)
            ordered.append(name)
            if len(ordered) >= top_k:
                break
        if len(ordered) >= top_k:
            break

    return [(name, float(len(ordered) - index)) for index, name in enumerate(ordered)]


def _base_rrf(base_arms: Mapping[str, Any], query: str, top_k: int) -> list[tuple[str, float]]:
    """The comparator leg: the ``rrf`` arm when built, else vector+fts fused."""
    if "rrf" in base_arms:
        return list(base_arms["rrf"](query, top_k))
    legs = [base_arms[name](query, top_k) for name in ("vector", "fts") if name in base_arms]
    if not legs:
        return []
    fused = _fuse(*legs, top_k=top_k)
    return fused


def _fuse(*rankings: Sequence[tuple[str, float]], top_k: int) -> list[tuple[str, float]]:
    """Reciprocal rank fusion at the registered constant (k=60)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return ordered[:top_k]


# ---------------------------------------------------------------------------
# Gate statistics and outcome mapping
# ---------------------------------------------------------------------------


def metric_result(
    deltas: Sequence[float],
    *,
    threshold: float,
    clusters: Sequence[str] | None = None,
) -> MetricResult:
    """Paired mean delta with a 95% percentile bootstrap interval.

    The resample order is fixed by ``BOOTSTRAP_SEED`` and the resampling uses
    one ``random.Random`` instance, so the same deltas produce the same bounds
    -- byte-for-byte -- on any machine and in any process (G1).

    With ``clusters`` (D-8) the resampling unit is the evidence cluster:
    clusters are drawn with replacement and *every* query in a drawn cluster
    enters the resample. Two questions about one document pair are not
    independent evidence, and treating them as independent understates the
    interval. Without ``clusters`` the per-query bootstrap is the
    anti-conservative secondary figure, never the primary one.

    At least two distinct clusters are required: a single cluster makes every
    resample identical, which would report a zero-width interval as if the
    effect were known exactly. The resampled statistic is the mean over every
    drawn query, which assumes equal cluster sizes; the committed bench has
    them (2 per multi-hop pair, 4 per aggregation group).
    """
    values = [float(delta) for delta in deltas]
    if not values:
        raise ValueError("no per-query deltas: a paired comparison needs at least one")
    point = math.fsum(values) / len(values)

    generator = random.Random(BOOTSTRAP_SEED)
    if clusters is None:
        count = len(values)
        means = [
            math.fsum(values[generator.randrange(count)] for _ in range(count)) / count
            for _ in range(BOOTSTRAP_RESAMPLES)
        ]
    else:
        if len(clusters) != len(values):
            raise ValueError(
                f"{len(clusters)} cluster ids for {len(values)} deltas: "
                "one id per query"
            )
        groups: dict[str, list[float]] = {}
        for cluster, value in zip(clusters, values, strict=True):
            if not isinstance(cluster, str) or not cluster:
                raise ValueError(
                    f"cluster ids must be non-empty strings, got {cluster!r}"
                )
            groups.setdefault(cluster, []).append(value)
        if len(groups) < 2:
            raise ValueError(
                "the cluster bootstrap needs at least two distinct clusters; "
                f"got {len(groups)}"
            )
        # The resampled statistic is the mean over every drawn query, exactly
        # as the pre-registration words it (equal cluster sizes in the bench).
        members = list(groups.values())
        means = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            drawn = [members[generator.randrange(len(members))] for _ in members]
            sample = [value for group in drawn for value in group]
            means.append(math.fsum(sample) / len(sample))

    means.sort()
    return MetricResult(
        point=point,
        ci_lower=_percentile(means, 0.025),
        ci_upper=_percentile(means, 0.975),
        threshold=threshold,
    )


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Nearest-rank percentile: the value at rank ``ceil(q * n)`` (1-based).

    ``n = 10_000`` and ``q = 0.025`` therefore take the 250th smallest value
    (index 249), and ``q = 0.975`` the 9,750th (index 9,749). This is the
    method the pre-registration fixes; it is not interpolated.
    """
    if not sorted_values:
        raise ValueError("no values to take a percentile of")
    rank = max(1, math.ceil(quantile * len(sorted_values)))
    return float(sorted_values[rank - 1])


def metric_outcome(result: MetricResult) -> str:
    """One metric's outcome under the registered mapping.

    NOT MET needs the *upper* bound below the threshold: that is the statement
    "this effect size is ruled out", which is stronger than "we did not see
    it". MET needs the point estimate at or above the threshold **and** the
    lower bound above zero. Everything between is INCONCLUSIVE.
    """
    if result.ci_upper < result.threshold:
        return OUTCOME_NOT_MET
    if result.point >= result.threshold and result.ci_lower > 0.0:
        return OUTCOME_MET
    return OUTCOME_INCONCLUSIVE


def evaluate_gate(
    *,
    ndcg: MetricResult,
    set_recall: MetricResult,
    regression_ok: bool,
    gate_arm: str = GATE_ARM,
) -> str:
    """The gate's outcome from its two metrics, the regression check, and arm.

    The 1-hop fallback is asymmetric on purpose: a NOT MET on multi-hop from an
    arm that only does one hop is recorded as INCONCLUSIVE, because it is not a
    fair test. MET is still MET, and a regression failure is still NOT MET.
    """
    if not regression_ok:
        return OUTCOME_NOT_MET

    outcomes = (metric_outcome(ndcg), metric_outcome(set_recall))
    if gate_arm == FALLBACK_GATE_ARM:
        if OUTCOME_NOT_MET in outcomes:
            return OUTCOME_INCONCLUSIVE
        return OUTCOME_MET if all(o == OUTCOME_MET for o in outcomes) else OUTCOME_INCONCLUSIVE

    if OUTCOME_NOT_MET in outcomes:
        return OUTCOME_NOT_MET
    if all(o == OUTCOME_MET for o in outcomes):
        return OUTCOME_MET
    return OUTCOME_INCONCLUSIVE


# ---------------------------------------------------------------------------
# Fidelity row (declared non-gating)
# ---------------------------------------------------------------------------


def fidelity_row(fixture_dir: Path | str, gold_path: Path | str) -> dict[str, Any]:
    """Extraction fidelity against the hand-verified gold subgraph.

    Reports, over the whole gold file: the fraction of gold entity names present
    after case-folding, the fraction of gold fact texts present, and (for
    multi-hop queries) the fraction whose joining entity exists and reaches all
    of the query's required documents through its extracted facts.

    ``gating`` is ``False`` and ``evaluate_gate`` takes nothing from this row:
    the row diagnoses extractor weakness, it never decides the gate.
    """
    records = load_graph_fixture(fixture_dir)
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    queries = gold.get("queries") or {}

    entity_names = {
        str(record["row"].get("name", "")).casefold()
        for record in records
        if record["kind"] == "node" and record["table"] == "Entity"
    }
    fact_texts = {
        str(record["row"].get("fact", "")).casefold().strip()
        for record in records
        if record["kind"] == "node" and record["table"] == "RelatesToNode_"
    }
    episode_names = {
        str(record["row"].get("uuid")): str(record["row"].get("name", ""))
        for record in records
        if record["kind"] == "node" and record["table"] == "Episodic"
    }

    entity_by_name: dict[str, str] = {}
    for record in records:
        if record["kind"] == "node" and record["table"] == "Entity":
            entity_by_name[str(record["row"].get("name", "")).casefold()] = str(
                record["row"].get("uuid")
            )

    # What a joining entity connects: the documents that mention it, plus the
    # documents reachable through the facts it takes part in -- the fact's own
    # episode and the documents mentioning the entity at its far end. That is
    # what "the entity connects two documents" means in a graph the extractor
    # built; a name that merely exists in one episode joins nothing.
    mentions: dict[str, set[str]] = {}
    for record in records:
        if record["kind"] == "rel" and record["table"] == "MENTIONS":
            mentions.setdefault(str(record["dst"]), set()).add(
                episode_names.get(str(record["src"]), "")
            )

    fact_episodes: dict[str, set[str]] = {}
    for record in records:
        if record["kind"] == "node" and record["table"] == "RelatesToNode_":
            fact_episodes[str(record["row"].get("uuid"))] = {
                episode_names.get(str(uuid), "")
                for uuid in record["row"].get("episodes") or []
            }

    fact_ends: dict[str, tuple[str, str]] = {}
    for record in records:
        if record["kind"] != "rel" or record["table"] != "RELATES_TO":
            continue
        if record["src_label"] == "Entity" and record["dst_label"] == "RelatesToNode_":
            fact_ends[str(record["dst"])] = (str(record["src"]), "")
        elif record["src_label"] == "RelatesToNode_" and record["dst_label"] == "Entity":
            source, _ = fact_ends.get(str(record["src"]), ("", ""))
            fact_ends[str(record["src"])] = (source, str(record["dst"]))

    entity_docs: dict[str, set[str]] = {
        entity_uuid: set(docs) for entity_uuid, docs in mentions.items()
    }
    for fact_uuid, (source, target) in fact_ends.items():
        episodes = fact_episodes.get(fact_uuid, set())
        for entity_uuid, other in ((source, target), (target, source)):
            if not entity_uuid:
                continue
            entity_docs.setdefault(entity_uuid, set()).update(episodes)
            entity_docs[entity_uuid].update(mentions.get(other, set()))

    gold_entities = {
        str(name).casefold()
        for query in queries.values()
        for name in query.get("entities") or []
    }
    gold_facts = {
        str(fact).casefold().strip()
        for query in queries.values()
        for fact in query.get("facts") or []
    }

    present_entities = {name for name in gold_entities if name in entity_names}
    missing_entities = sorted(gold_entities - present_entities)
    present_facts = {fact for fact in gold_facts if fact in fact_texts}

    multi_hop = [
        query
        for query in queries.values()
        if query.get("category") == "multi-hop" and query.get("joining_entity")
    ]
    connected = 0
    for query in multi_hop:
        joining = str(query["joining_entity"]).casefold()
        joining_uuid = entity_by_name.get(joining)
        required = {str(doc) for doc in query.get("documents") or []}
        if (
            joining_uuid is not None
            and required
            and required <= entity_docs.get(joining_uuid, set())
        ):
            connected += 1

    return {
        "entities_present": (
            len(present_entities) / len(gold_entities) if gold_entities else 1.0
        ),
        "missing_entities": missing_entities,
        "missing_facts": sorted(gold_facts - present_facts),
        "facts_present": (
            len(present_facts) / len(gold_facts) if gold_facts else 1.0
        ),
        "joins_connected": (connected / len(multi_hop)) if multi_hop else 1.0,
        "multi_hop_queries": len(multi_hop),
        "gating": False,
    }


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CATEGORY_FLAG_FLOOR",
    "EDGE_CAP",
    "FALLBACK_GATE_ARM",
    "FIXTURE_FORMAT",
    "GATE_ARM",
    "GRAPH_ARM_NAMES",
    "GRAPH_GROUP_ID",
    "GraphArmStatus",
    "GraphArmUnavailable",
    "GraphFixtureError",
    "MetricResult",
    "NDCG_THRESHOLD",
    "OUTCOME_INCONCLUSIVE",
    "OUTCOME_MET",
    "OUTCOME_NOT_MET",
    "REGRESSION_FLOOR",
    "RRF_K",
    "SEED_CAP",
    "SET_RECALL_THRESHOLD",
    "build_graph_arms",
    "evaluate_gate",
    "export_graph_fixture",
    "fidelity_row",
    "graph_arm_statuses",
    "import_graph_fixture",
    "load_fixture_manifest",
    "load_graph_fixture",
    "metric_outcome",
    "metric_result",
    "verify_fixture_manifest",
]
