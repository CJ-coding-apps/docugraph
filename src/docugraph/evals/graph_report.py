"""The 0.5.0 graph-bench report: every arm, the pre-registered gate, diagnostics.

``python -m docugraph.evals.graph_report --output <path>``

This is bench tooling, not a CLI command. It runs the five base arms and the
five graph arms over every query, reads the multi-hop gate exactly as
``docs/graph-bench-preregistration-2026-11-13.md`` fixes it, and writes one
JSON artifact -- the committed baseline (WP §4.6).

Two properties matter for the artifact:

* **No LLM is reachable.** The graph store is opened with an LLM client that
  raises if anything generates, and the fixture is imported from committed
  text. Query-time embeddings come from the frozen local embedder.
* **The report is deterministic.** It holds no wall-clock field and no
  absolute path, so two runs over the same corpus and fixture are
  byte-identical -- the check WP §6 asks CI to make. The baseline is
  identified by its content and the commit that introduced it.

Everything descriptive (fidelity row, per-query diagnostics, shape and
aggregation breakdowns) is marked non-gating; ``evaluate_gate`` receives only
the two registered multi-hop intervals and the regression check.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from docugraph.evals.arms import Arm, build_arms
from docugraph.evals.dataset import (
    MULTIHOP_SHAPES,
    SUPPORT_SET_CATEGORIES,
    BenchQuery,
    default_dataset_dir,
    load_corpus,
    load_manifest,
    load_queries,
    validate,
)
from docugraph.evals.graph_bench import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CATEGORY_FLAG_FLOOR,
    FALLBACK_GATE_ARM,
    GATE_ARM,
    NDCG_THRESHOLD,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_MET,
    REGRESSION_FLOOR,
    SEED_CAP,
    SET_RECALL_THRESHOLD,
    GraphFixtureError,
    build_graph_arms,
    entity_linking_recall,
    evaluate_gate,
    fidelity_per_query,
    fidelity_row,
    graph_arm_statuses,
    import_graph_fixture,
    load_fixture_manifest,
    metric_outcome,
    metric_result,
    verify_fixture_manifest,
)
from docugraph.evals.metrics import full_support_at_k, macro_average, set_recall_at_k
from docugraph.evals.runner import TOP_K, _environment, ingest_corpus, run_arm

GRAPH_REPORT_SCHEMA = "docugraph-graph-bench/1"
COMPARATOR = "rrf"

NDCG_METRIC = f"ndcg@{TOP_K}"
SET_METRIC = f"set_recall@{TOP_K}"
FULL_METRIC = f"full_support@{TOP_K}"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    REPO_ROOT / "benchmarks" / "retrieval" / "baselines" / "bge-small-en-v1.5-graph.json"
)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_arm(arm: Arm, queries: Sequence[BenchQuery]) -> dict[str, Any]:
    """Run one arm through the shared runner and add the set metrics.

    ``run_arm`` already scores recall/MRR/nDCG and keeps the top-k list the set
    metrics need; adding them here rather than in the runner leaves the
    existing benchmark's report schema untouched.
    """
    query_list = list(queries)
    report = run_arm(arm, query_list)
    rows = report["per_query"]
    for row, query in zip(rows, query_list, strict=True):
        required = query.required or ()
        row[SET_METRIC] = set_recall_at_k(row["top"], required, TOP_K)
        row[FULL_METRIC] = full_support_at_k(row["top"], required, TOP_K)
        row["cluster"] = query.cluster
        row["multihop_shape"] = query.multihop_shape

    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]] = [
        (report["overall"], rows),
    ]
    for category, aggregate in report["by_category"].items():
        groups.append((aggregate, [row for row in rows if row["category"] == category]))
    for aggregate, selected in groups:
        aggregate[SET_METRIC] = macro_average([row[SET_METRIC] for row in selected])
        aggregate[FULL_METRIC] = macro_average([row[FULL_METRIC] for row in selected])
    return report


def _rows(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["query_id"]: row for row in report["per_query"]}


def _deltas(
    reports: Mapping[str, Mapping[str, Any]],
    queries: Sequence[BenchQuery],
    metric: str,
    gate_arm: str,
) -> list[float]:
    gate = _rows(reports[gate_arm])
    comparator = _rows(reports[COMPARATOR])
    return [
        float(gate[query.id][metric]) - float(comparator[query.id][metric]) for query in queries
    ]


def _per_query_deltas(
    reports: Mapping[str, Mapping[str, Any]],
    queries: Sequence[BenchQuery],
    metric: str,
    gate_arm: str,
) -> dict[str, float]:
    gate = _rows(reports[gate_arm])
    comparator = _rows(reports[COMPARATOR])
    return {
        query.id: float(gate[query.id][metric]) - float(comparator[query.id][metric])
        for query in queries
    }


def _interval(
    deltas: Sequence[float], threshold: float, *, clusters: Sequence[str] | None
) -> dict[str, Any]:
    result = metric_result(deltas, threshold=threshold, clusters=clusters)
    return asdict(result)


def _secondary(deltas: Sequence[float], threshold: float) -> dict[str, Any]:
    return {
        **_interval(deltas, threshold, clusters=None),
        "label": "anti-conservative",
        "notes": "per-query resampling; not the primary interval (D-8)",
    }


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def regression_section(
    reports: Mapping[str, Mapping[str, Any]],
    queries: Sequence[BenchQuery],
    *,
    gate_arm: str = GATE_ARM,
) -> dict[str, Any]:
    """The within-run regression check on the existing single-chunk queries.

    The overall nDCG@10 difference must be at least ``REGRESSION_FLOOR``; each
    category also gets a per-query interval, and one whose upper bound lies
    entirely below ``CATEGORY_FLAG_FLOOR`` is flagged. A flag does not by
    itself fail the check -- the outcome mapping makes a MET gate INCONCLUSIVE
    until the flag is explained -- but it is reported, not hidden.
    """
    existing = [query for query in queries if query.category not in SUPPORT_SET_CATEGORIES]
    deltas = _deltas(reports, existing, NDCG_METRIC, gate_arm)
    overall = math.fsum(deltas) / len(deltas)

    by_category: dict[str, Any] = {}
    flagged: list[str] = []
    for category in sorted({query.category for query in existing}):
        selected = [query for query in existing if query.category == category]
        category_deltas = _deltas(reports, selected, NDCG_METRIC, gate_arm)
        result = metric_result(category_deltas, threshold=CATEGORY_FLAG_FLOOR)
        flag = result.ci_upper < CATEGORY_FLAG_FLOOR
        by_category[category] = {
            **asdict(result),
            "queries": len(selected),
            "flagged": flag,
        }
        if flag:
            flagged.append(category)

    return {
        "queries": len(existing),
        "overall_delta": overall,
        "floor": REGRESSION_FLOOR,
        "passed": overall >= REGRESSION_FLOOR,
        "category_flag_floor": CATEGORY_FLAG_FLOOR,
        "by_category": by_category,
        "flagged_categories": flagged,
        "gating": True,
    }


def gate_section(
    reports: Mapping[str, Mapping[str, Any]],
    queries: Sequence[BenchQuery],
    *,
    gate_arm: str = GATE_ARM,
) -> dict[str, Any]:
    """The pre-registered multi-hop gate, primary clustered and secondary per-query."""
    multi_hop = [query for query in queries if query.category == "multi-hop"]
    clusters = [str(query.cluster) for query in multi_hop]

    ndcg_deltas = _deltas(reports, multi_hop, NDCG_METRIC, gate_arm)
    set_deltas = _deltas(reports, multi_hop, SET_METRIC, gate_arm)

    ndcg_primary = metric_result(ndcg_deltas, threshold=NDCG_THRESHOLD, clusters=clusters)
    set_primary = metric_result(set_deltas, threshold=SET_RECALL_THRESHOLD, clusters=clusters)
    regression = regression_section(reports, queries, gate_arm=gate_arm)

    outcome = evaluate_gate(
        ndcg=ndcg_primary,
        set_recall=set_primary,
        regression_ok=regression["passed"],
        gate_arm=gate_arm,
    )
    # §5's mapping: a flagged category leaves a MET gate INCONCLUSIVE until it
    # is explained. It cannot turn NOT MET into anything weaker.
    if outcome == OUTCOME_MET and regression["flagged_categories"]:
        outcome = OUTCOME_INCONCLUSIVE

    return {
        "gate_arm": gate_arm,
        "comparator": COMPARATOR,
        "queries": len(multi_hop),
        "clusters": len(set(clusters)),
        "thresholds": {NDCG_METRIC: NDCG_THRESHOLD, SET_METRIC: SET_RECALL_THRESHOLD},
        NDCG_METRIC: {
            "primary_cluster": asdict(ndcg_primary),
            "primary_outcome": metric_outcome(ndcg_primary),
            "secondary_per_query": _secondary(ndcg_deltas, NDCG_THRESHOLD),
            "per_query_deltas": _per_query_deltas(reports, multi_hop, NDCG_METRIC, gate_arm),
        },
        SET_METRIC: {
            "primary_cluster": asdict(set_primary),
            "primary_outcome": metric_outcome(set_primary),
            "secondary_per_query": _secondary(set_deltas, SET_RECALL_THRESHOLD),
            "per_query_deltas": _per_query_deltas(reports, multi_hop, SET_METRIC, gate_arm),
        },
        "regression": regression,
        "outcome": outcome,
    }


def aggregation_section(
    reports: Mapping[str, Mapping[str, Any]],
    queries: Sequence[BenchQuery],
    *,
    gate_arm: str = GATE_ARM,
) -> dict[str, Any]:
    """The aggregation comparison: same statistics, no threshold, never gating."""
    groups = [query for query in queries if query.category == "aggregation"]
    clusters = [str(query.cluster) for query in groups]
    section: dict[str, Any] = {
        "gating": False,
        "threshold_applies": False,
        "queries": len(groups),
        "clusters": len(set(clusters)),
        "notes": "reported for the D2 decision; it cannot change the gate outcome",
    }
    for metric, threshold in (
        (NDCG_METRIC, NDCG_THRESHOLD),
        (SET_METRIC, SET_RECALL_THRESHOLD),
    ):
        deltas = _deltas(reports, groups, metric, gate_arm)
        section[metric] = {
            "cluster_bootstrap": _interval(deltas, threshold, clusters=clusters),
            "secondary_per_query": _secondary(deltas, threshold),
            "per_query_deltas": _per_query_deltas(reports, groups, metric, gate_arm),
        }
    return section


def shape_breakdown(
    reports: Mapping[str, Mapping[str, Any]],
    queries: Sequence[BenchQuery],
    *,
    gate_arm: str = GATE_ARM,
) -> dict[str, Any]:
    """Pre-registered descriptive split of the gate metrics by evidence shape.

    Fixed at C4, before any arm ran: bridge has two clusters and its interval
    will be very wide, which is reported as found. No threshold applies and the
    outcome is computed over all 32 multi-hop queries, never over this split.
    """
    multi_hop = [query for query in queries if query.category == "multi-hop"]
    section: dict[str, Any] = {}
    for shape in MULTIHOP_SHAPES:
        group = [query for query in multi_hop if query.multihop_shape == shape]
        clusters = [str(query.cluster) for query in group]
        entry: dict[str, Any] = {
            "gating": False,
            "threshold_applies": False,
            "queries": len(group),
            "clusters": len(set(clusters)),
        }
        if not group:
            section[shape] = entry
            continue
        for metric, threshold in (
            (NDCG_METRIC, NDCG_THRESHOLD),
            (SET_METRIC, SET_RECALL_THRESHOLD),
        ):
            deltas = _deltas(reports, group, metric, gate_arm)
            entry[metric] = {
                "cluster_bootstrap": (
                    _interval(deltas, threshold, clusters=clusters)
                    if len(set(clusters)) >= 2
                    else None
                ),
                "secondary_per_query": _secondary(deltas, threshold),
            }
        section[shape] = entry
    return section


def fixture_stats(fixture_dir: Path | str) -> dict[str, int]:
    """Counts a reader needs to interpret the fidelity row (descriptive)."""
    from docugraph.evals.graph_bench import GRAPH_GROUP_ID, load_graph_fixture

    records = load_graph_fixture(fixture_dir)
    episodes = {
        str(record["row"]["uuid"]): str(record["row"].get("name", ""))
        for record in records
        if record["kind"] == "node" and record["table"] == "Episodic"
    }
    entities = [
        record
        for record in records
        if record["kind"] == "node"
        and record["table"] == "Entity"
        and record["row"].get("group_id") == GRAPH_GROUP_ID
    ]
    facts = [
        record
        for record in records
        if record["kind"] == "node" and record["table"] == "RelatesToNode_"
    ]

    single_document = 0
    for fact in facts:
        documents = {episodes.get(str(uuid), "") for uuid in fact["row"].get("episodes") or []}
        if len(documents) == 1:
            single_document += 1

    entity_documents: dict[str, set[str]] = {}
    for record in records:
        if record["kind"] == "rel" and record["table"] == "MENTIONS":
            entity_uuid = str(record["dst"])
            entity_documents.setdefault(entity_uuid, set()).add(str(record["src"]))
    shared = sum(1 for documents in entity_documents.values() if len(documents) > 1)

    return {
        "episodes": len(episodes),
        "entities": len(entities),
        "facts": len(facts),
        "facts_with_one_source_document": single_document,
        "facts_with_two_source_documents": len(facts) - single_document,
        "entities_shared_across_documents": shared,
    }


def diagnostics_section(
    reports: Mapping[str, Mapping[str, Any]],
    queries: Sequence[BenchQuery],
    *,
    gate_arm: str,
    fixture_dir: Path,
    gold_path: Path,
    embedder: Any,
) -> dict[str, Any]:
    """Per-multi-hop-query evidence for diagnosing a NOT MET or INCONCLUSIVE."""
    multi_hop = [query for query in queries if query.category == "multi-hop"]
    gate = _rows(reports[gate_arm])
    comparator = _rows(reports[COMPARATOR])
    per_query_fidelity = fidelity_per_query(fixture_dir, gold_path)
    linking = entity_linking_recall(
        fixture_dir, embedder=embedder, queries=multi_hop, gold_path=gold_path
    )

    rows = []
    for query in multi_hop:
        rows.append(
            {
                "query_id": query.id,
                "cluster": query.cluster,
                "shape": query.multihop_shape,
                "required": list(query.required or ()),
                f"gate_{NDCG_METRIC}": gate[query.id][NDCG_METRIC],
                f"rrf_{NDCG_METRIC}": comparator[query.id][NDCG_METRIC],
                f"gate_{SET_METRIC}": gate[query.id][SET_METRIC],
                f"rrf_{SET_METRIC}": comparator[query.id][SET_METRIC],
                "gate_top": gate[query.id]["top"],
                "rrf_top": comparator[query.id]["top"],
                "fidelity": per_query_fidelity.get(query.id),
                "entity_linking": linking.get(query.id),
            }
        )

    return {
        "gating": False,
        "fidelity_row": fidelity_row(fixture_dir, gold_path),
        "fixture_stats": fixture_stats(fixture_dir),
        "per_query": rows,
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def raising_llm() -> Any:
    """A Graphiti LLM client whose every generation path raises.

    The fixture is imported from text and every arm is LLM-free; attaching this
    client turns "no LLM at query time" into a runtime fact instead of an
    assumption.
    """
    from graphiti_core.llm_client.client import LLMClient
    from graphiti_core.llm_client.config import LLMConfig

    class RaisingLLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(config=LLMConfig(model="bench", small_model="bench"))

        async def _generate_response(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("LLM used at query time")

        async def generate_response(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("LLM used at query time")

    return RaisingLLM()


def _version_of(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except Exception:
        return None


def report_environment(
    vector_store: Any,
    dataset_manifest: Mapping[str, Any],
    fixture_manifest: Mapping[str, Any],
    *,
    gate_arm: str | None,
) -> dict[str, Any]:
    """Everything needed to compare this run to another (WP §4.6, Phase 0 block)."""
    environment = _environment(vector_store, dict(dataset_manifest))
    environment.update(
        {
            "python": platform.python_version(),
            "graphiti": fixture_manifest.get("graphiti_version"),
            "kuzu": _version_of("kuzu"),
            "gate_arm": gate_arm,
            "comparator": COMPARATOR,
            "thresholds": {
                NDCG_METRIC: NDCG_THRESHOLD,
                SET_METRIC: SET_RECALL_THRESHOLD,
            },
            "regression_floor": REGRESSION_FLOOR,
            "category_flag_floor": CATEGORY_FLAG_FLOOR,
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "percentile": "nearest-rank",
                "primary_unit": "evidence-cluster",
                "secondary_unit": "query",
            },
            "seed_cap": SEED_CAP,
            "fixture": {
                "format": fixture_manifest.get("fixture"),
                "group_id": fixture_manifest.get("group_id"),
                "ingest_reference_time": fixture_manifest.get("ingest_reference_time"),
                "llm_provider": fixture_manifest.get("llm_provider"),
                "llm_model": fixture_manifest.get("llm_model"),
                "prompt_sha256": fixture_manifest.get("prompt_sha256"),
                "extraction_git_sha": fixture_manifest.get("extraction_git_sha"),
                "embedder": fixture_manifest.get("embedder"),
                "sha256": fixture_manifest.get("sha256"),
            },
        }
    )
    return environment


def run_graph_bench(
    dataset_dir: Path,
    store_dir: Path,
    graph_db_path: Path,
    *,
    fixture_dir: Path | None = None,
    embedder: Any | None = None,
) -> dict[str, Any]:
    """Run every arm, judge the gate, and return the baseline report dict.

    Raises when the dataset does not validate or the fixture does not verify:
    numbers over either are not comparable to anything.
    """
    problems = validate(dataset_dir)
    if problems:
        raise ValueError("dataset is not sound:\n" + "\n".join(f"- {p}" for p in problems))

    fixture = Path(fixture_dir) if fixture_dir is not None else dataset_dir / "graph_fixture"
    fixture_problems = verify_fixture_manifest(fixture)
    if fixture_problems:
        raise GraphFixtureError(
            "the graph fixture does not verify:\n"
            + "\n".join(f"- {problem}" for problem in fixture_problems)
        )

    corpus = load_corpus(dataset_dir)
    queries = load_queries(dataset_dir)
    manifest = load_manifest(dataset_dir)
    fixture_manifest = load_fixture_manifest(fixture)

    if embedder is None:
        from docugraph.core.embeddings import get_embedder

        embedder = get_embedder()

    from docugraph.storage.vector_store import VectorStore

    vector_store = VectorStore(db_path=store_dir, embedder=embedder)
    ingest_corpus(vector_store, corpus)
    base_arms = build_arms(vector_store)

    reports: dict[str, Any] = {
        name: score_arm(arm, queries) for name, arm in sorted(base_arms.items())
    }
    statuses: dict[str, dict[str, Any]] = {
        name: {"available": True, "reason": None} for name in sorted(base_arms)
    }
    graph_statuses = graph_arm_statuses(fixture_manifest=fixture_manifest, embedder=embedder)
    statuses.update(
        {
            name: {"available": status.available, "reason": status.reason}
            for name, status in sorted(graph_statuses.items())
        }
    )

    unavailable = [
        status.reason
        for status in graph_statuses.values()
        if not status.available and status.reason
    ]
    # If the graph arms cannot run, the report still holds the non-graph arms
    # and an INCONCLUSIVE outcome with the reason -- never a silent omission.
    graph_error: str | None = unavailable[0] if unavailable else None

    if graph_error is None:
        from docugraph.storage.graph_store import GraphStore

        store = GraphStore(graph_db_path, llm_client=raising_llm(), embedder=embedder)
        try:
            asyncio.run(import_graph_fixture(store, fixture))
            graph_arms = build_graph_arms(
                store=store,
                embedder=embedder,
                fixture_manifest=fixture_manifest,
                base_arms=base_arms,
                doc_ids={doc.id for doc in corpus},
            )
            for name, arm in sorted(graph_arms.items()):
                reports[name] = score_arm(arm, queries)
        finally:
            asyncio.run(store.close())

    if GATE_ARM in reports:
        gate_arm: str | None = GATE_ARM
    elif FALLBACK_GATE_ARM in reports:
        gate_arm = FALLBACK_GATE_ARM
    else:
        gate_arm = None

    if gate_arm is None:
        gate: dict[str, Any] = {
            "outcome": OUTCOME_INCONCLUSIVE,
            "reason": graph_error or "no graph arm was available",
            "gating": True,
        }
    else:
        gate = gate_section(reports, queries, gate_arm=gate_arm)

    diagnostics: dict[str, Any]
    if gate_arm is None:
        diagnostics = {
            "gating": False,
            "fidelity_row": fidelity_row(fixture, dataset_dir / "gold_subgraph.json"),
            "fixture_stats": fixture_stats(fixture),
            "per_query": [],
        }
    else:
        diagnostics = diagnostics_section(
            reports,
            queries,
            gate_arm=gate_arm,
            fixture_dir=fixture,
            gold_path=dataset_dir / "gold_subgraph.json",
            embedder=embedder,
        )

    report: dict[str, Any] = {
        "schema": GRAPH_REPORT_SCHEMA,
        "top_k": TOP_K,
        "corpus_sha256": manifest["sha256"]["corpus.jsonl"],
        "queries_sha256": manifest["sha256"]["queries.jsonl"],
        "fixture_verified": [],
        "environment": report_environment(
            vector_store, manifest, fixture_manifest, gate_arm=gate_arm
        ),
        "arm_statuses": statuses,
        "arms": reports,
        "gate": gate,
        "aggregation": (
            aggregation_section(reports, queries, gate_arm=gate_arm)
            if gate_arm is not None
            else {"gating": False, "available": False}
        ),
        "shape_breakdown": (
            shape_breakdown(reports, queries, gate_arm=gate_arm)
            if gate_arm is not None
            else {"gating": False, "available": False}
        ),
        "diagnostics": diagnostics,
    }
    return report


# ---------------------------------------------------------------------------
# CLI (bench tooling, not a docugraph command)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m docugraph.evals.graph_report",
        description="Run every bench arm and write the graph-bench baseline report.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset_dir(),
        help="benchmark dataset directory (default: the committed dataset)",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="graph fixture directory (default: <dataset>/graph_fixture)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"report path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="docugraph-graph-bench-") as work:
        report = run_graph_bench(
            args.dataset,
            Path(work) / "vector",
            Path(work) / "graph",
            fixture_dir=args.fixture,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gate = report["gate"]
    print(f"gate: {gate['outcome']} (arm {gate.get('gate_arm')!r})")
    if "ndcg@10" in gate:
        for metric in (NDCG_METRIC, SET_METRIC):
            primary = gate[metric]["primary_cluster"]
            print(
                f"  {metric}: {primary['point']:+.4f} "
                f"[{primary['ci_lower']:+.4f}, {primary['ci_upper']:+.4f}] "
                f"threshold {primary['threshold']:+.2f} "
                f"({gate[metric]['primary_outcome']})"
            )
        regression = gate["regression"]
        print(
            f"  regression: {regression['overall_delta']:+.4f} "
            f"(floor {regression['floor']:+.2f}, passed={regression['passed']})"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
