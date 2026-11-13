# WP-docugraph-0.5.0 — Measure the graph before building on it

**Repository:** CJ-coding-apps/docugraph · **Local checkout:** `/Users/james/projects/docugraph-ai-v1` (**not** `docugraph-ai` or `docugraph-local`, which are old v0.1.0 trees) · **Base:** `main` @ 6a7afa9 (v0.4.0) · **Branch:** `feat/0.5.0-graph-bench`
**Revision 2** (2026-11-13): §5 gate comparator pinned to `rrf`; gate arm named, with a pre-registered outcome mapping; P0-3 and P0-5 sharpened; `group_ids` at search; test N1 and R1 added.
**Issued by:** CJ (maintainer), 2026-11-13 · **Reviewer:** an independent verification session (Claude), per AGENTS.md §8.4
**Governing rules:** CJ's AGENTS.md constitution, revision 2. Where this order and the constitution disagree, the constitution wins (§11), and you report the conflict.
**Background:**
- The evaluation `docugraph_graph_retrieval_evaluation_2026-11-13.md` lives in CJ's claude.ai Project, not on disk; its conclusions are restated in §1.
- The bench spec (2026-10-23) and the bench as built: `benchmarks/retrieval/` and `src/docugraph/evals/`.

---

## 0. How this order runs
1. Research, then plan (assumptions first), then **STOP** for review.
2. **Phase 0** (determinism), then the gate in §3.
3. Tests first: blind, committed alone, and shown red on 6a7afa9.
4. Implement.
5. Verify on CJ's Mac and in CI, naming the environment for every result.
6. Open the PR. **Do not merge.** Merges, tags, releases and fixture regeneration are CJ's acts.

**Reserved-file exceptions (this PR only):** `.github/workflows/ci.yml`, only to add the fixture-integrity and graph-bench steps in §6; existing job names unchanged. Nothing else reserved. **No dependency changes:** if one seems necessary, stop and ask.

---

## 1. Why
The bench has five arms (`vector`, `fts`, `lancedb-hybrid`, `rrf`, `weighted`) and seven single-chunk query categories. The graph leg is unmeasured, and none of the categories is one where a graph should help.

**Goal:** add the categories where a graph should earn its cost (multi-hop and aggregation), measure graph arms on them deterministically, and record whether a pre-registered gate is met.

The release does **not** build a new graph schema. That decision (D2: build on graphiti or own the schema) is taken afterwards, with this data.

**Calibration:** in the committed baseline, hybrid beats the best single leg by about +0.02–0.03 nDCG@10. That's the realistic prior; the gate in §5 is designed not to fire on noise.

---

## 2. Maintainer decisions (D-2026-11-13-#; CJ confirms before handover)
1. **Measure, don't build.** No schema change, no Claim or Summary nodes, no LLM at query time, no ACL or tenancy.
2. *(Deferred)* Graph model: build on graphiti vs own schema. Decided after this release's results.
3. **The fixture is regenerated only by CJ,** with an explicit command that needs his API key. Its manifest is **load-bearing**: CI fails if the fixture's content hash, or any recorded version in it, doesn't match.
4. **The queries are written by hand.** The agent drafts the queries, design notes, qrels and gold subgraph. **They are frozen only after the independent review session and CJ have both approved the design notes.**
5. **The gate is pre-registered (§5)** and committed to the spec *before* any graph arm is run on the bench. The comparator is fixed as `rrf`, and the gate arm is fixed by the Phase 0 result, as §5 sets out.
6. **Namespacing:** graphiti's `group_id` is the namespace field. The fixture is ingested with `group_id = "bench"`, and every graph arm searches with `group_ids=["bench"]` (`GraphStore.search`, store.py:530). No ID-format change.

---

## 3. Phase 0 — is graphiti search deterministic over a frozen graph? (before the tests-first commit)
Run on CJ's Mac, against a small graph built once from five or more bench documents with a real LLM. CJ provides the key for this one run, or names a provider already configured.
- **P0-1 · No LLM in the query path.** Trace every graphiti search recipe the graph arms will use (entity-seeded, fact/edge, BFS 2-hop) and every call they reach. Assert at runtime that the LLM client is never called: use a client that raises on any call.
- **P0-2 · Identity reranker and frozen embedder, asserted.** At search time, assert that the cross-encoder is docugraph's `NoopCrossEncoder` and that the embedder is the local fastembed model named in config (id and dimension). A config path that silently picks a cloud embedder must fail this check, not pass a grep.
- **P0-3 · No wall-clock dependence.** Determine whether any recipe filters edges against the current time when no as-of is given. A static read of graphiti 0.30.2 suggests the query path makes no `utc_now()` call (they are all in ingest and maintenance), and that the `search()` docstring's "current date and time as the reference point" is stale, so this is expected to pass. Confirm it by trace, not by the docstring. If a time filter is found, the arms must pass a fixed reference time. Either way, keep the test that runs search with the clock patched to two dates a year apart and requires identical output; it's the cheap proof that nothing downstream reads the clock.
- **P0-4 · Byte-identical across processes.** Run the same queries in **three separate processes with different `PYTHONHASHSEED`**, serialise the ranked results (IDs, scores rounded to 1e-6, ranks), and require them to be byte-identical. A same-process rerun isn't enough.
- **P0-5 · Fixture format.** Find a text, diffable export of the graphiti graph that round-trips. Candidates: Kùzu `EXPORT DATABASE` to CSV, or a graphiti-level JSON dump. It must re-import and give identical P0-4 output, so the bench can rebuild the graph from committed text in CI with no LLM. **The export must carry the node and edge embeddings and the temporal fields** (`valid_at`, `invalid_at`, `expired_at`, `created_at`). Nothing is re-embedded or recomputed at import; a re-embed would quietly change rankings. Test R1 checks this field by field.

**Gate:**
- **All five pass:** continue straight into tests-first, without stopping again; post the Phase 0 results in the PR.
- **P0-1, P0-3 or P0-4 fails:** stop and report. A deterministic query-time linker then becomes a separate work order.
- **P0-5 fails:** stop and report the options.

---

## 4. Behavior

### 4.1 New query categories
- **multi-hop:** at least **16 queries**. Each answer needs two or more documents joined by a named entity. Each `design_note` names the joining entity and the required documents.
- **aggregation:** at least **16 queries**. Each answer is a **support set**: the N atomic facts or chunks that must all be retrieved. Each `design_note` lists the support set.
- New documents are added to the fictional `qrv` corpus, with stable human-readable IDs, as the existing corpus does. Existing documents and queries are untouched.
- `MANIFEST.json` gains the new files, with their hashes, through the existing integrity check.

### 4.2 Set-level metrics (in `src/docugraph/evals/metrics.py`)
- `set_recall@k`: the fraction of the required set that appears in the top k.
- `full_support@k`: 1 if the *entire* required set is in the top k, else 0.
- These are reported for every arm, with per-query detail, alongside the existing metrics, which are unchanged. On the existing seven categories, set recall equals today's recall, and a test proves that.

### 4.3 The gold subgraph (extraction-fidelity control)
`benchmarks/retrieval/gold_subgraph.json`, hand-verified. For each multi-hop and aggregation query it records the required documents, the joining or aggregated entities, and the required facts.

**The fidelity row** reports, for the fixture: the fraction of gold entities present (after name normalisation), the fraction of gold facts present as edges, and the fraction of multi-hop queries whose joining entity exists and connects the required documents. It is **declared non-gating**, and its purpose is to tell retriever weakness apart from extractor weakness.

### 4.4 The graph fixture (D-3)
- **Builder:** `python -m docugraph.evals.build_graph_fixture`. It is not a CLI command; it's bench tooling. It refuses to run without an explicit `--i-am-regenerating` flag and a configured LLM key.
- It ingests the bench corpus through docugraph's own `GraphStore` (`group_id = "bench"`), exports in the P0-5 format to `benchmarks/retrieval/graph_fixture/`, and writes `graph_fixture/MANIFEST.json`, recording:
  - the content hash
  - the LLM model ID
  - the hash of the graphiti prompt sources
  - the graphiti version
  - the git SHA of docugraph's extraction code
  - the embedder model and its dimension
  - the ingest reference time: the time used for `valid_at` defaults during extraction. It is a fixture property, not a query-time parameter, unless P0-3 finds a query-time time filter.
- **The bench** rebuilds the graph from the committed export (no LLM), then verifies the manifest.

### 4.5 Graph arms (precise definitions; each reports a status)
- **`graph-entity`:** "graph expansion seeded by entity vector match". Seeds are entities matched by query embedding; results are the chunks or episodes reached within 1 hop. It also reports **entity-linking recall**: whether the gold joining entity is among the seeds.
- **`graph-fact`** (diagnostic): fact/edge retrieval by edge embedding plus BM25, using graphiti's edge search. Results are mapped to their source chunks. This tests whether multi-hop failures come from the level the graph works at (entities vs facts).
- **`graph-2hop`:** like `graph-entity`, with a 2-hop BFS and a fan-out cap. Run only if P0-1 and P0-4 hold for that recipe; otherwise mark it unavailable, with the reason. **The fan-out cap is a number fixed in the plan,** before the gate is committed, not tuned on results.
- **`rrf+graph-2hop`:** RRF over `vector`, `fts` and `graph-2hop`. **The primary gate arm** (§5).
- **`rrf+graph`:** RRF over `vector`, `fts` and `graph-entity`. Diagnostic, or the gate arm only if `graph-2hop` is unavailable.
- **RRF settings:** the fused arms use the same RRF constant and per-leg depth as the existing `rrf` arm, so the only difference from the comparator is the added leg.
- **Namespace:** every graph arm calls search with `group_ids=["bench"]`; test N1 proves it.
- **Mapping:** results from every graph arm are mapped to chunk IDs through episode provenance, so all arms are scored on the same qrels. The mapping is documented and tested.

### 4.6 Baseline
After the gate is registered (§5), run every arm on all categories and commit the new baseline JSON, including the fidelity row and the Phase 0 environment block (model versions, reference time, seeds).

---

## 5. Pre-registered gate (committed to the spec before any graph arm runs)
**Comparator:** `rrf` (vector + fts), the 0.4.0 winner, fixed in advance. Comparing against "the best non-graph arm" on the same run would pick the comparator from the data: the CI would no longer be a valid 95% interval, and a different run could name a different "best". Differences against the other non-graph arms are reported descriptively only.

**Gate arm:** `rrf+graph-2hop` if Phase 0 shows `graph-2hop` available; otherwise `rrf+graph` (1-hop). The plan records which, from the Phase 0 results, before the gate is committed.

**Statistics:** for each gate metric, d = gate arm − `rrf` per query, on **multi-hop**. Paired bootstrap over queries, 10,000 resamples, fixed seed, 95% percentile CI. The thresholds are **+0.05 nDCG@10** and **+0.10 set_recall@10**. Both metrics must pass (an intersection test), so no multiplicity correction is applied.

**Regression check (within this run, on the new corpus):** on the existing 56 queries, the gate arm's overall nDCG@10 minus `rrf`'s must be ≥ **−0.02**. Per-category differences are reported with CIs. Any category whose CI lies entirely below −0.05 is **flagged**. Adding documents changes `corpus_sha256`, so the report compares the two arms within this run. It doesn't present the existing categories' numbers as unchanged from the 0.4.0 baseline, and it states the corpus hash of both.

**Outcome mapping (pre-registered):**
- **MET:** for both metrics, the point estimate is ≥ its threshold and the CI lower bound is > 0. The overall regression check passes and no category is flagged.
- **NOT MET:** for at least one metric, the CI upper bound is below its threshold (the effect size is ruled out); or the overall regression check fails. *Exception:* if the gate arm is the 1-hop fallback, a NOT MET on the multi-hop metrics is recorded as **INCONCLUSIVE** (a 1-hop arm isn't a fair test of multi-hop).
- **INCONCLUSIVE:** everything else, including the CI containing 0 while the point estimate clears the threshold, an unavailable gate arm, and a flagged category (until explained).

With 16 queries the intervals will be wide, and INCONCLUSIVE is a likely outcome. The report says so plainly and includes, per multi-hop query, entity-linking recall and fidelity-row coverage, so that a NOT MET or INCONCLUSIVE can be diagnosed (retriever vs extractor), not just recorded.

**Aggregation** is reported with the same statistics against `rrf`, but is not part of the gate. It informs D2: summary nodes vs graphiti communities.

**Outcome:** the PR states MET, NOT MET or INCONCLUSIVE, with the numbers. Nothing is built on the outcome in this PR.

---

## 6. CI
- **The existing bench job** gains: the fixture-manifest verification (it fails on a mismatch), the graph arms run from the committed export, and a check that the run's output is byte-identical to a second run in a fresh process with a different `PYTHONHASHSEED`.
- **No LLM key in CI**, ever.

## 7. Non-goals (binding)
- Schema changes, Claim or Summary nodes, entity-resolution changes.
- LLM reranking, or any LLM use at query time.
- ACL or tenancy.
- A Ladybug-backend graph bench (Kùzu default only).
- Batch-API extraction.
- Any MCP or CLI behaviour change.

## 8. Worked examples (each a test)
| # | Case | Expected |
|---|---|---|
| D1 | the three-process `PYTHONHASHSEED` run over the fixture | byte-identical ranked output |
| D2 | the LLM client is replaced by one that raises; run every graph arm | no call |
| D3 | config selects a cloud embedder | the graph arms refuse with a status, never run silently |
| D4 | the clock patched ±1 year | identical output |
| F1 | the fixture content is altered by one byte | the manifest check fails |
| F2 | a manifest graphiti version that differs from the installed one | the check fails, naming both |
| M1 | `set_recall` on single-chunk queries | equals today's recall on all 56 |
| M2 | `full_support@10` with one of two required chunks found | 0; `set_recall@10` = 0.5 |
| G1 | the bootstrap with a fixed seed | reproducible CI bounds |
| G2 | the gate evaluated on synthetic arms (one clearly better, one noise) | MET for the better one, NOT MET or INCONCLUSIVE for noise |
| G3 | the outcome mapping on synthetic deltas: (a) CI above 0 and point ≥ threshold; (b) CI upper < threshold; (c) CI spans 0, point ≥ threshold; (d) 1-hop fallback with (b) | MET; NOT MET; INCONCLUSIVE; INCONCLUSIVE |
| P1 | the fidelity row on a fixture missing one gold entity | reports the miss; the gate is unaffected |
| N1 | the fixture plus a decoy episode under `group_id = "other"` that matches a query strongly | no graph arm returns it |
| R1 | export → import round-trip | node/edge embeddings and temporal fields equal field by field; P0-4 output identical |

## 9. Mutation checks (show each goes red)
- Remove the manifest check → F1 red.
- Let one recipe call the LLM → D2 red.
- Sort with an unstable tie-break → D1 red.
- Drop the reference time → D4 red (if P0-3 found a time filter).
- Drop `group_ids` from a graph arm's search → N1 red.
- Re-embed at import → R1 red.
- Choose the comparator as the max over non-graph arms → G3 red.

## 10. Release note
This work is bench and eval tooling. If no user-facing behaviour changes, a PyPI release may not be warranted. **Whether to tag 0.5.0 and publish is CJ's call** after the PR merges.
