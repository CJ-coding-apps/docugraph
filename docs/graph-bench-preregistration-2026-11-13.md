# Graph bench — pre-registration (C0)

**Date:** 2026-11-13 · **Work order:** `WP-docugraph-0.5.0-graph-bench.md` (revision 2) §4.5, §5
**Phase 0:** `docs/phase0-determinism-results-2026-11-13.md` — all five checks PASS
**Base:** `6a7afa9` (v0.4.0) + `090ef87` (work order) + `b8bbfad` (extraction-client fix)

This document is committed **before any graph arm has been run on the bench**.
It fixes the rules and the numbers. Nothing here may be changed after the first
bench run; a later change is a new pre-registration with its own commit and its
own reason, stated before the new numbers are seen.

The *queries* are not fixed here. They are drafted in C3 and frozen in C4, and
only after an independent review session and CJ have approved the design notes
(binding condition). This document fixes how the arms search and how the gate
is judged, not what is asked.

---

## 1. Gate

### Comparator

> **`rrf`** — the existing vector + FTS RRF arm, the 0.4.0 winner, fixed in
> advance.

No other non-graph arm is a comparator. Differences against them are reported
descriptively only. This prevents the comparator from being chosen by the data
(which would invalidate the interval).

### Gate arm

> **`rrf+graph-2hop`**

Phase 0 measured `graph-2hop` available: the recipe touches no LLM (P0-1), is
byte-identical across processes (P0-4), and its expansion reads only the frozen
embedder and the graph driver. §5's fallback to `rrf+graph` therefore does not
apply. The 1-hop arm remains a diagnostic and is still run.

### Statistics

For each gate metric, per query: `d = gate_arm − rrf`, over **multi-hop queries
only**. Paired bootstrap over queries, **10,000 resamples**, **seed `20261113`**,
95% percentile CI.

Thresholds — both must pass (intersection-union test, no multiplicity
correction):

| Metric | Threshold |
|---|---|
| `ndcg@10` | **+0.05** |
| `set_recall@10` | **+0.10** |

### Regression check (same run, existing 56 queries)

- overall `ndcg@10` difference (gate − `rrf`) must be **≥ −0.02**;
- per-category differences reported with CIs; a category whose CI lies entirely
  below **−0.05** is **flagged**.

Adding documents changes `corpus_sha256`. Both arms are compared *within this
run*; the report states the corpus hash and does not present existing-category
numbers as unchanged from the 0.4.0 baseline.

### Outcome mapping (fixed)

- **MET** — both metrics: point estimate ≥ threshold **and** CI lower bound > 0;
  regression check passes; no category flagged.
- **NOT MET** — at least one metric's CI upper bound is below its threshold; or
  the regression check fails. *Exception:* with the 1-hop fallback as gate arm,
  NOT MET on multi-hop is recorded **INCONCLUSIVE**. (Not applicable here.)
- **INCONCLUSIVE** — everything else, including a CI containing 0 while the
  point estimate clears the threshold, an unavailable gate arm, and a flagged
  category until it is explained.

With 16–20 queries the interval will be wide; INCONCLUSIVE is a likely outcome
and must be reported as such, not reframed.

Aggregation queries get the same statistics against `rrf` and are **not** part
of the gate.

---

## 2. Arm rules (fixed numbers)

All values below were measured on the Phase 0 probe graph and are fixed now. They
are not tuned on bench results.

### Entity-seeded expansion (`graph-entity`, `graph-2hop`)

- Query vector: the frozen `BAAI/bge-small-en-v1.5` query embedding (with the
  bge query prefix, as `FastEmbedEmbedder.embed_query` already does).
- Entity score: cosine(query vector, `Entity.name_embedding`).
- **Seeds (distance 0): top `SEED_CAP = 20` entities** by `(−cosine, uuid)`
  — UUID ascending breaks ties.
- **Distance 1 chunks:** episodes that `MENTIONS` a seed entity in the top 20.
- **Distance 2 entities:** the far-end entity of each **kept fact** incident to
  a seed entity.
- **Kept facts: at most `EDGE_CAP = 50` per hop**, ranked by
  cosine(query vector, `RelatesToNode_.fact_embedding`), ties by fact UUID
  ascending.
- **Distance 2 chunks:** episodes that `MENTIONS` a distance-2 entity.
- A fact's own episode is **not** admitted at distance 2: it belongs to the
  seed's document and is already covered at distance 1.
- `graph-entity` = the same procedure with `hops = 1` (no distance-2 step).

### Chunk ranking (both arms)

Deduplicate chunks; a chunk's key is its **best** offering —
minimum distance, then maximum seed score, then maximum edge similarity. Final
order:

> **`(distance, −seed score, −edge similarity, chunk_id)`**

`seed score` = cosine of the seed entity that reached the chunk; `edge
similarity` = cosine of the fact that reached it (0.0 at distance 1);
`chunk_id` = corpus document id, ascending.

### `graph-fact` (diagnostic only)

Graphiti `EDGE_HYBRID_SEARCH_RRF` with `limit = 10` against the same graph;
edges mapped to chunks through their episodes. Results map into the common
chunk space; no fan-out parameters of its own.

### Namespace

Every graph arm calls search with `group_ids=["bench"]`. This is verified by
test N1, not assumed.

### Fusion

Fused arms (`rrf+graph-2hop`, `rrf+graph`) use the **same RRF constant and leg
depth as `rrf`**: `rrf_k = 60` (`HybridRetriever` config), `top_k = 10`. The
only difference from the comparator is the added leg. Depth per leg is the
arm's requested `top_k`; the graph leg returns at most that many chunks after
its own ranking (the expansion may reach more; it is truncated to `top_k`
before fusion).

### Mapping to chunks/units of relevance

- `runner.ingest_corpus` loads **one document = one chunk**, `chunk.id ==
  doc.id`, `document_id == doc.id` (verified in `src/docugraph/evals/runner.py:41`).
- The graph fixture is built with episode `name = doc.id`; the mapping
  episode → chunk is therefore `Episodic.name` → chunk id.
- A test proves the mapping (an episode that does not resolve to a known doc id
  is an error, not a silent drop).

### Set metrics

`set_recall@k` = fraction of the required set in the top k;
`full_support@k` = 1 iff the entire required set is in the top k. Reported for
every arm and every category. On the existing seven categories a test proves
`set_recall` equals today's `recall`.

---

## 3. Fixture (fixed format)

- Directory: `benchmarks/retrieval/graph_fixture/`
- `graph.jsonl` — `{"kind":"node"|"rel", ...}` records, floats by `json`
  repr (exact float32), timestamps ISO-8601.
- `schema.cypher` — Kùzu's own DDL.
- `MANIFEST.json` — content hash; LLM model id; graphiti prompt-source hash;
  graphiti version; git SHA of the extraction code; embedder model and
  dimension; ingest reference time; group id `bench`.
- Nothing is re-embedded or recomputed at import. FTS indexes are recreated by
  `_ensure_fts_indexes`.
- `MANIFEST.json` also notes that the extraction model is a remotely served
  `:cloud` tag. The **committed fixture** is the reproducible artifact; a
  regeneration under the same tag is not guaranteed to produce the same graph,
  because the tag can point at different weights or serving behaviour over time.
  The manifest's content hash identifies what is committed, not what a rebuild
  would yield.
- The query embedder's id and dimension must equal the manifest's, or the graph
  arms **refuse** with a status (R-1). Same dimension with a different id is also
  refused.

---

## 4. Extraction model (D-7)

> **Ollama**, model `deepseek-v4.1-flash:cloud`, temperature `0.0`,
> ingest reference time `2026-01-01T00:00:00Z`.

Recorded in the fixture manifest. No LLM key exists in CI; CI never rebuilds the
fixture.

---

## 5. Anti-tuning clause

After C5 (CJ regenerates the fixture) and the first gate run:

- `SEED_CAP`, `EDGE_CAP`, hop count, the ranking key, `rrf_k`, the comparator,
  the metrics, the thresholds, the bootstrap seed and the outcome mapping are
  **frozen**.
- If the gate returns NOT MET or INCONCLUSIVE, the PR reports it and the
  diagnosis (entity-linking recall, fidelity row) — it does not re-tune the arm
  and re-run.
- Any diagnostic change (e.g. a different fan-out) is a separate, later
  experiment with its own pre-registration.

The one exception is §6. It is a fixture-quality decision taken on the frozen
corpus before any arm is run, so it cannot be tuned on performance.

---

## 6. One-time extractor check (CJ-approved 2026-11-13)

Phase 0 found zero cross-document entity links on a probe corpus that does
share three domain terms across documents (results §6.1). A fixture that sparse
would make the bench measure the extractor, not graph retrieval — and the later
D2 decision on graphiti versus an own schema would rest on it. This check
decides the extraction model once, before any results exist.

Sequence:

1. C4 freezes the corpus; CJ builds the fixture (C5).
2. Compute **only the fidelity row** (WP §4.3). No graph arm is run, no qrels
   are scored, no retrieval metric is computed.
3. The deciding quantity is the fidelity row's multi-hop component: the
   fraction of multi-hop queries whose joining entity exists in the fixture
   (after name normalisation) **and** connects the required documents.
4. If that fraction is **< 0.50**, CJ may switch the extraction model **once**,
   before C6. The replacement model id is recorded before the second fidelity
   row is computed; CJ rebuilds the fixture with it; the fidelity row is
   computed again. Nothing is ranked in either pass.
5. Record in the PR body: both fidelity rows, both fixture manifest content
   hashes, both model ids, the commit of the superseded fixture, and the
   reason. If the first coverage is ≥ 0.50, the fixture stands and C6 proceeds.
6. If the second coverage is still < 0.50, the bench proceeds as pre-registered
   and reports it; there is no third model.

The superseded fixture's manifest and commit remain in git history; the
recorded content hashes identify both.

---

## 7. Not fixed here

- The multi-hop and aggregation queries and their qrels (C3, frozen at C4 after
  independent review + CJ approval).
- The gold subgraph contents (C3).
- Whether CI runs D1/D4 on the real fixture or the synthetic one (R-1; both are
  acceptable to this pre-registration, and the choice does not affect the
  gate's numbers).
