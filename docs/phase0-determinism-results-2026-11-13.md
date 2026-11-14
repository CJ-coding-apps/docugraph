# Phase 0 results — is graphiti search deterministic over a frozen graph?

**Date:** 2026-11-13 · **Work order:** `WP-docugraph-0.5.0-graph-bench.md` (revision 2), §3
**Ran on:** CJ's Mac (Darwin 24.5.0, x86_64) · **Result:** **all five checks PASS**
**Graph built with:** D-7 = Ollama provider, model `deepseek-v4.1-flash:cloud`
**Extraction-code SHA:** `b8bbfad1e1daac6e82898e5270ec3189707629ea`

This document is the Phase 0 record. It is committed before any graph arm is run
on the bench (C0 pre-registration) and before the acceptance tests are written.
The probe source is Appendix A.

---

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.11.13 |
| graphiti-core | 0.30.2 |
| kuzu | 0.11.3 |
| fastembed | 0.8.0 |
| lancedb (not used by the graph fixture) | 0.25.3 |
| Embedder | `BAAI/bge-small-en-v1.5`, 384 dimensions |
| Cross-encoder | docugraph `NoopCrossEncoder` (identity order) |
| Extraction LLM (D-7) | Ollama · `deepseek-v4.1-flash:cloud` |
| Extraction temperature | `0.0` |
| Ingest reference time | `2026-01-01T00:00:00Z` |
| Group namespace | `bench` |

`Entity.created_at`, `RelatesToNode_.created_at` and `Episodic.created_at` are
wall-clock ingest times. They are a fixture property: captured once at build
time, exported with the fixture, and never recomputed.

## 2. The graph that was built

Ten documents from `benchmarks/retrieval/corpus.jsonl`, ingested through
`GraphStore.add_episode` with `group_id="bench"` and episode `name` = corpus
document id:

`qrv-architecture`, `qrv-guide-checkpointing`, `qrv-observability`,
`qrv-sink-postgres`, `qrv-err-4012`, `qrv-troubleshooting`,
`qrv-err-index`, `qrv-sinks-overview`, `qrv-sources-overview`,
`qrv-transforms`.

| Table | Rows |
|---|---|
| `Episodic` | 10 |
| `Entity` | 33 |
| `RelatesToNode_` (facts) | 5 |
| `MENTIONS` | 33 |
| `RELATES_TO` | 10 |
| `Community`, `Saga`, HAS_* | 0 |

Per-document extraction (entities / new facts): architecture 1/0,
checkpointing 3/1, observability 4/1, sink-postgres 1/0, err-4012 0/0,
troubleshooting 1/0, err-index 12/0, sinks-overview 1/0, sources-overview 8/2,
transforms 2/1.

## 3. Results

### P0-1 · No LLM in the query path — PASS

The run replaced both `graphiti.llm_client` and `graphiti.clients.llm_client`
with an object that raises on **any attribute access**, then ran, for ten
queries: `search_()` with `EDGE_HYBRID_SEARCH_RRF`, `search_()` with
`NODE_HYBRID_SEARCH_RRF`, and the entity-seeded 1-hop and 2-hop expansion
(direct table reads: `Entity.name_embedding`, `RELATES_TO`,
`RelatesToNode_.fact_embedding`, `Episodic`/`MENTIONS`). No access occurred;
the run completed and produced results.

Static confirmation: graphiti 0.30.2's `graphiti_core/search/` imports no clock
and no LLM entry point. `Graphiti.search()` uses `EDGE_HYBRID_SEARCH_RRF`.

### P0-2 · Identity reranker and frozen embedder — PASS

Asserted at runtime on the constructed store, not by inspecting config:

- `type(graphiti.cross_encoder) is NoopCrossEncoder`
- `isinstance(graphiti.embedder, GraphitiEmbedderAdapter)`
- the wrapped docugraph embedder reports `model_name == "BAAI/bge-small-en-v1.5"`
- its measured `dimensions == 384`

The assertion ran on every probe run, including the clock and hash-seed runs.

### P0-3 · No wall-clock dependence — PASS

Static trace of `graphiti_core/search/*.py`: the only clock-shaped calls are
`time()` for latency measurement (`search.py:109,246`,
`search_utils.py:1220,1243,1907,1934`); none feeds a ranking, filter or
returned value. `utc_now` appears only in ingest and maintenance
(`node_operations`, `edge_operations`, `community_operations`,
`combined_extraction`, `nodes`, `graphiti.add_episode`). The `Graphiti.search`
docstring's "current date and time as the reference point for temporal
relevance" is stale: no such filter exists.

Runtime proof: 18 clock bindings reachable from the graph stack (`utc_now`
names and `from datetime import datetime` bindings in every loaded
`graphiti_core` module) were patched to `2025-01-01T00:00:00` and separately to
`2027-01-01T00:00:00`. Both outputs are byte-identical to the unpatched run.

### P0-4 · Byte-identical across processes — PASS

Ten queries, four recipes, ranked ids with scores rounded to 1e-6, serialised
as canonical JSON (`sort_keys=True`, compact separators). Three separate
processes with `PYTHONHASHSEED=0`, `1`, `2`:

```
9c54c1d38cfc58c094ca44a2852be36b856f4ad935958f05addbad1399e3eef8
```

all three. This includes graphiti's RRF reranker and the custom expansion with
its explicit `(-score, uuid)` tie-breaks.

### P0-5 · Fixture format — PASS (JSONL, not CSV)

**Kùzu CSV is supported but unusable.** `EXPORT DATABASE '<dir>' (format='csv')`
works (statement syntax; `CALL export_database(...)` does not exist), but the
writer formats floats with six decimals. Measured: `0.123456789` →
`0.123457`, and `1.23e-07` → `0.000000`. P0-5 requires float round-trip, so CSV
is ruled out. Parquet is exact but not text/diffable.

**Chosen format:** a graphiti-level text dump, `graph_fixture/`:

- `schema.cypher` — the DDL Kùzu itself emits (node tables, rel tables,
  tokenizer macros), 2,840 bytes, sha256 `88272198adb367559236038896a9a3509198cd287b4edb43501777db77fed5cf`
- `graph.jsonl` — one JSON object per line, either
  `{"kind":"node","table":T,"row":{...}}` or
  `{"kind":"rel","table":T,"src_label":L,"src":U,"dst_label":M,"dst":V,"row":{...}}`.
  Floats are written by `json` (shortest repr → exact), timestamps as ISO-8601.
  91 records, 360,883 bytes, sha256 `fb18c4b5cbd85c50e774eef510ae31c6f9f50821d76ba784da62db38bf08ab24`

`graph.jsonl` carries every column, including `name_embedding`,
`fact_embedding`, `valid_at`, `invalid_at`, `expired_at`, `created_at` and
`reference_time`. Nothing is re-embedded or recomputed at import: embeddings are
bound as parameters, and the FTS indexes are recreated by the store's existing
`_ensure_fts_indexes` (they are not data).

Round-trip evidence:

- import of the 91 records into a fresh Kùzu database succeeds;
- the same ten-query probe against the imported graph has the same sha256
  `9c54c1d3…` as the original;
- field-by-field comparison of all 91 records: **0 mismatches**, embeddings
  (33 entity + 5 fact vectors) equal, temporal fields equal, rel endpoints
  equal.

## 4. Finding that changes extraction code (separate commit `b8bbfad`)

Graphiti's resilient client was constructed in its default `json_schema` mode.
That mode delegates enforcement to the server. Ollama's OpenAI-compatible
endpoint ignores `response_format` for `deepseek-v4.1-flash:cloud`, and the
native `/api/chat` `format` field is ignored too (measured: prose response; and
for the extraction prompt a bare list of `{name, type}` with no
`entity_type_id`). Extraction therefore failed with a pydantic
`ValidationError` before any node was written.

The fix is the setting graphiti documents for this case:
`structured_output_mode="json_object"`, in which `generate_response` appends
the response model's JSON schema to the last message. The docugraph client
overrides only the inner request loop, so the injection still runs. With that
one setting the same model emits `entity_type_id` and extraction succeeds.

This is committed separately (`b8bbfad`, `fix(llm): …`) because it is a
pre-existing extraction defect, not part of the bench. The extraction-code SHA
recorded in the fixture manifest will be a tree that includes it.

## 5. Gate-arm determination

`graph-2hop` is **available**: P0-1 and P0-4 hold for the recipe, and its
custom expansion reads only the driver and the frozen embedder. Per §5, the
gate arm is therefore:

> **`rrf+graph-2hop`** (not the 1-hop fallback).

## 6. Risk found — recorded, not a Phase 0 failure

The extractor produced **no cross-document entity links** on this graph: every
fact's two entities come from the same document, and no entity reached by a
fact is mentioned in a different document. Consequently the 2-hop arm returned
exactly the same chunks as the 1-hop arm for all ten queries (measured), and
the probe exercises the 2-hop code path but does not demonstrate additional
retrieval value.

This does not affect determinism (P0-1/P0-3/P0-4 are about the query path,
which is independent of extraction density). It is a fixture risk: if the
full-corpus fixture is equally sparse, the gate arm cannot win, and the
fidelity row will show why. The report must present this as extractor weakness,
not retriever weakness, and §5's INCONCLUSIVE is a likely outcome.

### 6.1 Measured: do the probe documents share entities?

Zero links has two possible causes — the documents share no entities, or they
share entities the extractor missed. Measured on the built graph:

- every entity node is mentioned by exactly one episode (33 `MENTIONS`, 33
  entities; no entity with more than one episode, no name owned by two nodes);
- **three domain terms do appear in more than one of the ten documents**:

| term | probe documents |
|---|---|
| `QRV-4012` | `qrv-err-4012`, `qrv-err-index` |
| `qv.Sink` | `qrv-sink-postgres`, `qrv-sinks-overview` |
| `Quorva` | `qrv-architecture`, `qrv-guide-checkpointing`, `qrv-troubleshooting` |

In each case the graph holds the entity (or a name variant) from only one of
them: `QRV-4012` is extracted only from `qrv-err-index` (the err-4012 document
extracted 0 entities); `qv.Sink` only from `qrv-sinks-overview` (the
sink-postgres document extracted `qv.sinks.PostgresSink` instead); `quorva`
only from `qrv-troubleshooting`, while the architecture document has a distinct
node `Quorva pipeline`. The one further string overlap, `JSON`
(observability + sources-overview), is a generic term, not a joinable entity.

So the probe documents are **not entity-disjoint**, but with three shared terms
they are close to it, and every shared term is registered in only one document.
The missing cross-document links are therefore at least partly extraction misses
rather than an absence of shared entities — but three terms are too few to say
how a richer corpus behaves. That is why the real corpus gets its own
measurement: the one-time extractor check, pre-registered in
`docs/graph-bench-preregistration-2026-11-13.md` §6, run after the corpus is
frozen and before any arm is scored.

## 7. Reproduction

```
# build (real LLM; Appendix B) -- writes /tmp/phase0/graph
python /tmp/phase0/build2.py && python /tmp/phase0/add.py

# determinism
PYTHONHASHSEED=0 python probe.py search > a.json
PYTHONHASHSEED=1 python probe.py search > b.json
PYTHONHASHSEED=2 python probe.py search > c.json
shasum -a 256 a.json b.json c.json

# clock
python probe.py search --clock 2025 > d.json
python probe.py search --clock 2027 > e.json

# fixture
python probe.py export /tmp/fixture
python probe.py import /tmp/fixture /tmp/imported
python probe.py search --graph-dir /tmp/imported > f.json
```

The probe is scratch instrumentation and is not a module in the package
(decision Q5). The three-process run becomes the D1 acceptance test.

---

## Appendix A — probe source

The exact instrument used for the measurements above, verbatim. Paths point at
the scratch directory used on 2026-11-13. The graph-build scripts are in
Appendix B.

```python
"""Phase 0 probe for the docugraph 0.5.0 graph bench (not committed).

Modes:
  search [--clock YEAR]   run every recipe the graph arms will use and emit
                          canonical ranked JSON
  trace                   static/dynamic clock check
  export OUT              write the fixture as text (schema.cypher + graph.jsonl)
  import IN OUT           rebuild a graph DB from the fixture
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/Users/james/projects/docugraph-ai-v1")
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("DOCUGRAPH_STORAGE__DATA_DIR", "/tmp/phase0/data")
os.environ.setdefault("DOCUGRAPH_LLM__PROVIDER", "ollama")
os.environ.setdefault("DOCUGRAPH_LLM__MODEL", "deepseek-v4.1-flash:cloud")

GRAPH_DIR = Path("/tmp/phase0/graph")
DB_FILE = GRAPH_DIR / "graphiti.kuzu"

KS = (1, 5, 10)
FIXTURE_EMBEDDER_ID = "BAAI/bge-small-en-v1.5"
FIXTURE_EMBEDDER_DIM = 384
GROUP = "bench"
SEED_CAP = 20
EDGE_CAP = 50

QUERIES = [
    "checkpoint_store_uri configuration key",
    "default log_format value",
    "Prometheus metrics endpoint",
    "QRV-4012 sink write rejected",
    "PostgresSink flush interval",
    "Quorva pipeline stages",
    "state store durable storage",
    "structured JSON logs",
    "commit_interval_s setting",
    "troubleshooting stalled pipeline",
]

CANON_JSON = dict(sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class RaisingLLM:
    """Any attribute access is a failure: the query path must not touch it."""

    def __getattr__(self, name):
        raise AssertionError(f"LLM client used at query time (attribute {name!r})")


class FakeDatetime(datetime):
    _fixed = datetime(2026, 1, 1, 0, 0, 0)

    @classmethod
    def now(cls, tz=None):
        return cls._fixed if tz is None else cls._fixed.replace(tzinfo=tz)

    @classmethod
    def utcnow(cls):
        return cls._fixed


def patch_clocks(year: int) -> int:
    """Point every clock the graph path can reach at a fixed instant.

    Static trace: graphiti 0.30.2 imports no clock in ``search/``; all
    ``utc_now`` uses are in ingest/maintenance. This patches the names that do
    exist, so a clock read that appeared later would surface here too.
    """
    fixed = datetime(year, 1, 1, 0, 0, 0)
    FakeDatetime._fixed = fixed
    patched = 0
    for mod in list(sys.modules.values()):
        name = getattr(mod, "__name__", "")
        if not name.startswith("graphiti_core"):
            continue
        if getattr(mod, "utc_now", None) is not None:
            mod.utc_now = lambda: fixed
            patched += 1
        # Modules that did `from datetime import datetime` hold the class.
        if getattr(mod, "datetime", None) is datetime:
            mod.datetime = FakeDatetime
            patched += 1
    return patched


GRAPH_DIR_OVERRIDE = None


async def _store():
    from docugraph.storage.graph_store import GraphStore

    store = GraphStore(GRAPH_DIR_OVERRIDE or GRAPH_DIR)
    g = await store._ensure_initialized()
    return store, g


def _cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def _rows(driver, cypher: str):
    records, _header, _summary = await driver.execute_query(cypher)
    return records


async def load_graph(driver):
    """Read the tables the graph arms need, in a deterministic order."""
    ents = await _rows(
        driver,
        f"MATCH (n:Entity) WHERE n.group_id = '{GROUP}' "
        "RETURN n.uuid, n.name, n.name_embedding ORDER BY n.uuid",
    )
    facts = await _rows(
        driver,
        f"MATCH (a:Entity)-[r:RELATES_TO]->(e:RelatesToNode_) WHERE e.group_id = '{GROUP}' "
        "RETURN a.uuid, e.uuid, e.fact_embedding, e.episodes ORDER BY e.uuid, a.uuid",
    )
    # fact -> target entity (the reverse direction of the same edge)
    fact_targets = await _rows(
        driver,
        f"MATCH (e:RelatesToNode_)-[r:RELATES_TO]->(b:Entity) WHERE e.group_id = '{GROUP}' "
        "RETURN e.uuid, b.uuid ORDER BY e.uuid, b.uuid",
    )
    mentions = await _rows(
        driver,
        f"MATCH (ep:Episodic)-[r:MENTIONS]->(n:Entity) WHERE ep.group_id = '{GROUP}' "
        "RETURN ep.uuid, ep.name, n.uuid ORDER BY ep.uuid, n.uuid",
    )
    return ents, facts, fact_targets, mentions


def _get(row, *names):
    for n in names:
        if n in row:
            return row[n]
    raise KeyError(f"{names} not in {list(row)}")


async def custom_arm(g, query: str, hops: int) -> list[dict]:
    """Entity-seeded expansion with a pre-registered fan-out and ranking.

    Entity distance: seeds are 0. A fact incident to an entity at distance d
    puts the entity on its far end at distance d+1.
    Chunks: distance-1 chunks are episodes that mention a seed; distance-2
    chunks are episodes that mention an entity reached by a kept fact (the
    fact's own episode belongs to the seed's document and adds nothing new).

    Ranking (pre-registered): (distance, -seed score, -edge similarity,
    chunk_id). Edges kept: cosine of the fact embedding to the query, ties by
    UUID, at most EDGE_CAP per hop. Seeds: top SEED_CAP by (-cosine, uuid).
    """
    embedder = g.embedder
    qvec = await embedder.create(query)
    driver = g.driver
    ents, facts, fact_targets, mentions = await load_graph(driver)

    ent_rows = []
    for r in ents:
        emb = _get(r, "n.name_embedding", "name_embedding")
        ent_rows.append((_get(r, "n.uuid", "uuid"), _cos(qvec, list(emb))))
    ent_rows.sort(key=lambda t: (-t[1], t[0]))
    seeds = ent_rows[:SEED_CAP]
    seed_score = dict(seeds)

    mentions_by_entity: dict[str, list[str]] = {}
    for r in mentions:
        mentions_by_entity.setdefault(_get(r, "n.uuid", "uuid"), []).append(
            _get(r, "ep.uuid", "uuid")
        )

    chunk: dict[str, tuple[int, float, float]] = {}

    def offer(ep_uuid: str, cand: tuple[int, float, float]) -> None:
        cur = chunk.get(ep_uuid)
        key = (cand[0], -cand[1], -cand[2])
        if cur is None or key < (cur[0], -cur[1], -cur[2]):
            chunk[ep_uuid] = cand

    # distance 1: episodes that mention a seed entity
    for uuid, score in seeds:
        for ep_uuid in mentions_by_entity.get(uuid, ()):
            offer(ep_uuid, (1, score, 0.0))

    # distance 2: entities reached by a kept fact, then their episodes
    if hops >= 2:
        candidates = []
        for r in facts:
            a = _get(r, "a.uuid", "uuid")
            f = _get(r, "e.uuid", "uuid")
            if a in seed_score:
                sim = _cos(qvec, list(_get(r, "e.fact_embedding", "fact_embedding")))
                candidates.append((f, a, sim))
        candidates.sort(key=lambda t: (-t[2], t[0]))
        kept = candidates[:EDGE_CAP]
        best: dict[str, tuple[float, float]] = {}
        for f, a, sim in kept:
            for r in fact_targets:
                if _get(r, "e.uuid", "uuid") != f:
                    continue
                far = _get(r, "b.uuid", "uuid")
                pair = (seed_score[a], sim)
                if far not in best or (-pair[0], -pair[1]) < (-best[far][0], -best[far][1]):
                    best[far] = pair
        for far, (sscore, sim) in best.items():
            for ep_uuid in mentions_by_entity.get(far, ()):
                offer(ep_uuid, (2, sscore, sim))

    # map episode uuid -> doc id (episode name)
    eps = await _rows(
        driver,
        f"MATCH (e:Episodic) WHERE e.group_id = '{GROUP}' RETURN e.uuid, e.name ORDER BY e.uuid",
    )
    ep_name = {_get(r, "e.uuid", "uuid"): _get(r, "e.name", "name") for r in eps}

    ranked = []
    for ep_uuid, (hop, sscore, esim) in chunk.items():
        doc = ep_name.get(ep_uuid)
        if doc is None:
            continue
        ranked.append((doc, hop, sscore, esim))
    ranked.sort(key=lambda t: (t[1], -t[2], -t[3], t[0]))
    return [{"id": d, "rank": i + 1, "score": round(s, 6), "hop": h, "edge_sim": round(e, 6)}
            for i, (d, h, s, e) in enumerate(ranked)]


async def run_search(clock_year: int | None) -> dict:
    # Import the graph stack first: patch_clocks can only rewrite bindings that
    # already exist in sys.modules.
    import docugraph.storage.graph_store  # noqa: F401
    import graphiti_core  # noqa: F401
    import graphiti_core.graphiti  # noqa: F401
    import graphiti_core.nodes  # noqa: F401
    import graphiti_core.utils.datetime_utils  # noqa: F401

    if clock_year is not None:
        n = patch_clocks(clock_year)
        print(f"# patched {n} clock bindings to {clock_year}", file=sys.stderr)
    store, g = await _store()

    # P0-2: identity reranker and frozen embedder, asserted.
    from docugraph.storage.graph_store import GraphitiEmbedderAdapter, NoopCrossEncoder

    assert type(g.cross_encoder) is NoopCrossEncoder, type(g.cross_encoder)
    assert isinstance(g.embedder, GraphitiEmbedderAdapter), type(g.embedder)
    inner = g.embedder._embedder
    assert inner.model_name == FIXTURE_EMBEDDER_ID, inner.model_name
    assert inner.dimensions == FIXTURE_EMBEDDER_DIM, inner.dimensions

    # P0-1: the query path must never touch the LLM client.
    g.llm_client = RaisingLLM()
    g.clients.llm_client = RaisingLLM()  # type: ignore[union-attr]

    from graphiti_core.search.search_config_recipes import (
        EDGE_HYBRID_SEARCH_RRF,
        NODE_HYBRID_SEARCH_RRF,
    )

    out: dict[str, dict] = {}
    for q in QUERIES:
        ecfg = EDGE_HYBRID_SEARCH_RRF.model_copy(update={"limit": 10})
        ncfg = NODE_HYBRID_SEARCH_RRF.model_copy(update={"limit": 10})
        edges = await g.search_(q, config=ecfg, group_ids=[GROUP])
        nodes = await g.search_(q, config=ncfg, group_ids=[GROUP])
        esc = list(edges.edge_reranker_scores) + [None] * len(edges.edges)
        nsc = list(nodes.node_reranker_scores) + [None] * len(nodes.nodes)
        out[q] = {
            "edge-rrf": [{"id": e.uuid, "rank": i + 1,
                          "score": None if esc[i] is None else round(float(esc[i]), 6)}
                         for i, e in enumerate(edges.edges)],
            "node-rrf": [{"id": n.uuid, "name": n.name, "rank": i + 1,
                          "score": None if nsc[i] is None else round(float(nsc[i]), 6)}
                         for i, n in enumerate(nodes.nodes)],
            "entity-1hop": await custom_arm(g, q, 1),
            "entity-2hop": await custom_arm(g, q, 2),
        }
    await store.close()
    return out


async def export_fixture(out_dir: Path) -> None:
    import kuzu

    out_dir.mkdir(parents=True, exist_ok=True)
    db = kuzu.Database(str(DB_FILE))
    c = kuzu.Connection(db)
    # DDL from Kuzu itself (text, diffable); the data goes to JSON so floats
    # keep full precision -- the CSV writer rounds to 6 decimals (measured).
    import shutil
    tmp = out_dir / "_ddl"
    shutil.rmtree(tmp, ignore_errors=True)
    c.execute(f"EXPORT DATABASE '{tmp}' (format='csv')")
    (out_dir / "schema.cypher").write_text((tmp / "schema.cypher").read_text())

    tables = [r[0] for r in c.execute("CALL show_tables() RETURN name, type;").get_all()]
    kinds = {r[0]: r[1] for r in c.execute("CALL show_tables() RETURN name, type;").get_all()}
    lines = []
    for t in sorted(tables):
        if kinds[t] == "NODE":
            df = c.execute(f"MATCH (n:`{t}`) RETURN n.*;").get_as_df()
            for row in df.to_dict(orient="records"):
                row = {k.split(".", 1)[-1]: _jsonable(v) for k, v in row.items()}
                lines.append({"kind": "node", "table": t, "row": row})
        else:
            # endpoints are carried by column names a.uuid / b.uuid
            df = c.execute(
                f"MATCH (a)-[r:`{t}`]->(b) "
                "RETURN label(a) AS src_label, label(b) AS dst_label, "
                "a.uuid AS src, b.uuid AS dst, r.*;"
            ).get_as_df()
            for row in df.to_dict(orient="records"):
                src, dst = row.pop("src"), row.pop("dst")
                sl, dl = row.pop("src_label"), row.pop("dst_label")
                row = {k.split(".", 1)[-1]: _jsonable(v) for k, v in row.items()}
                lines.append({"kind": "rel", "table": t, "src": src, "dst": dst,
                              "src_label": sl, "dst_label": dl, "row": row})
    with (out_dir / "graph.jsonl").open("w") as fh:
        for line in lines:
            fh.write(json.dumps(line, sort_keys=True, ensure_ascii=True) + "\n")
    shutil.rmtree(tmp)
    print(f"wrote {out_dir} ({len(lines)} records)")


def _jsonable(v):
    import numpy as np
    import pandas as pd

    if v is None or v is pd.NaT:
        return None
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.to_pydatetime().isoformat()
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def _from_json(v):
    if v == "NaN":
        return float("nan")
    if isinstance(v, str) and len(v) >= 10 and v[4] == "-" and v[7] == "-":
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return v
    if isinstance(v, list):
        return [_from_json(x) for x in v]
    return v


def import_fixture(in_dir: Path, db_dir: Path) -> None:
    import shutil

    import kuzu

    shutil.rmtree(db_dir, ignore_errors=True)
    db_dir.mkdir(parents=True)
    db = kuzu.Database(str(db_dir / "graphiti.kuzu"))
    c = kuzu.Connection(db)
    ddl = (in_dir / "schema.cypher").read_text()
    for stmt in _split_cypher(ddl):
        c.execute(stmt)
    records = [json.loads(line) for line in (in_dir / "graph.jsonl").open()]
    # nodes first
    for rec in records:
        if rec["kind"] != "node":
            continue
        row = {k: _from_json(v) for k, v in rec["row"].items()}
        cols = ", ".join(f"`{k}`: ${k}" for k in row)
        c.execute(f"CREATE (:`{rec['table']}` {{{cols}}})", row)
    for rec in records:
        if rec["kind"] != "rel":
            continue
        row = {k: _from_json(v) for k, v in rec["row"].items()}
        cols = ", ".join(f"`{k}`: ${k}" for k in row)
        params = dict(row, src=rec["src"], dst=rec["dst"])
        c.execute(
            f"MATCH (a:`{rec['src_label']}` {{uuid: $src}}), "
            f"(b:`{rec['dst_label']}` {{uuid: $dst}}) "
            f"CREATE (a)-[:`{rec['table']}` {{{cols}}}]->(b)",
            params,
        )
    print(f"imported {len(records)} records into {db_dir}")


def _split_cypher(text: str) -> list[str]:
    """Split on semicolons outside quotes (the DDL has quoted regex text)."""
    out, buf, quote, escape = [], [], None, False
    for ch in text:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if quote:
            buf.append(ch)
            if ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "export", "import"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--clock", type=int)
    ap.add_argument("--graph-dir")
    args = ap.parse_args()
    global GRAPH_DIR_OVERRIDE
    GRAPH_DIR_OVERRIDE = args.graph_dir
    if args.mode == "search":
        print(json.dumps(asyncio.run(run_search(args.clock)), **CANON_JSON))
    elif args.mode == "export":
        asyncio.run(export_fixture(Path(args.paths[0])))
    else:
        import_fixture(Path(args.paths[0]), Path(args.paths[1]))


if __name__ == "__main__":
    main()
```


## Appendix B — graph-build scripts

Two scripts, because the first run stopped at the extraction defect described in
§4 and the remaining four documents were added after the fix. Both are verbatim.
The line `store._force_json_object = True` in the first script was a debug
leftover: `GraphStore` has no such attribute, so the assignment is inert. The
operative setting is `g.llm_client.structured_output_mode = "json_object"`,
which after `b8bbfad` is already the client's default.

```python
import asyncio, json, os, sys, time
from datetime import datetime, UTC
from pathlib import Path

REPO = Path("/Users/james/projects/docugraph-ai-v1")
sys.path.insert(0, str(REPO / "src"))
os.environ["DOCUGRAPH_STORAGE__DATA_DIR"] = "/tmp/phase0/data"
os.environ["DOCUGRAPH_LLM__PROVIDER"] = "ollama"
os.environ["DOCUGRAPH_LLM__MODEL"] = "deepseek-v4.1-flash:cloud"
os.environ["DOCUGRAPH_LLM__SMALL_MODEL"] = "deepseek-v4.1-flash:cloud"
os.environ["DOCUGRAPH_LLM__TEMPERATURE"] = "0"

from docugraph.storage.graph_store import GraphStore

DOCS = ["qrv-architecture", "qrv-guide-checkpointing", "qrv-observability",
        "qrv-sink-postgres", "qrv-err-4012", "qrv-troubleshooting"]
REF = datetime(2026, 1, 1, tzinfo=UTC)
db = Path("/tmp/phase0/graph")
if db.exists(): raise SystemExit("graph exists; remove it first")

corpus = {}
for line in open(REPO / "benchmarks/retrieval/corpus.jsonl"):
    d = json.loads(line); corpus[d["id"]] = d

async def main():
    store = GraphStore(db)
    store._force_json_object = True
    try:
        g = await store._ensure_initialized()
        g.llm_client.structured_output_mode = "json_object"
        for doc_id in DOCS:
            d = corpus[doc_id]
            t = time.time()
            r = await store.add_episode(
                name=doc_id, content=d["content"], source_type="text",
                source_description=d["title"], group_id="bench", reference_time=REF)
            print(f"{doc_id}: ep={r['episode_uuid']} ents={r['entities_count']} rels={r['relationships_count']} {time.time()-t:.1f}s", flush=True)
            print("   entities:", [e["name"] for e in r["entities"]], flush=True)
    finally:
        await store.close()

asyncio.run(main())
```

```python
import asyncio, json, os, sys
from datetime import datetime, UTC
from pathlib import Path
REPO=Path("/Users/james/projects/docugraph-ai-v1"); sys.path.insert(0,str(REPO/"src"))
os.environ.update({"DOCUGRAPH_STORAGE__DATA_DIR":"/tmp/phase0/data","DOCUGRAPH_LLM__PROVIDER":"ollama","DOCUGRAPH_LLM__MODEL":"deepseek-v4.1-flash:cloud","DOCUGRAPH_LLM__SMALL_MODEL":"deepseek-v4.1-flash:cloud","DOCUGRAPH_LLM__TEMPERATURE":"0"})
from docugraph.storage.graph_store import GraphStore
DOCS=["qrv-err-index","qrv-sinks-overview","qrv-sources-overview","qrv-transforms"]
corpus={json.loads(l)["id"]:json.loads(l) for l in open(REPO/"benchmarks/retrieval/corpus.jsonl")}
async def main():
    s=GraphStore(Path("/tmp/phase0/graph"))
    g=await s._ensure_initialized(); g.llm_client.structured_output_mode="json_object"
    try:
        for d in DOCS:
            r=await s.add_episode(name=d,content=corpus[d]["content"],source_type="text",
                source_description=corpus[d]["title"],group_id="bench",reference_time=datetime(2026,1,1,tzinfo=UTC))
            print(d, r["entities_count"], r["relationships_count"], [e["name"] for e in r["entities"]], flush=True)
    finally: await s.close()
asyncio.run(main())
```
