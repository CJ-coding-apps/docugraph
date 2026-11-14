"""C6 report assembly: the gate, its diagnostics, and the set-metric scoring.

The statistics themselves are tested in ``test_graph_bench_gate.py``. These
tests cover the wiring the report adds on top: which queries enter each
section, how the gate arm's outcome is read off the registered mapping, and
the two per-query diagnostics that let a miss be attributed to the retriever
or to the extractor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docugraph.evals.dataset import BenchQuery
from docugraph.evals.graph_bench import (
    GATE_ARM,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_MET,
    OUTCOME_NOT_MET,
)
from docugraph.evals.graph_report import (
    aggregation_section,
    gate_section,
    regression_section,
    score_arm,
    shape_breakdown,
)

from .graph_bench_helpers import KeywordEmbedder, build_synthetic_fixture


def _row(
    query_id: str,
    category: str,
    *,
    ndcg: float,
    set_recall: float,
    cluster: str | None = None,
    shape: str | None = None,
) -> dict:
    return {
        "query_id": query_id,
        "category": category,
        "ndcg@10": ndcg,
        "set_recall@10": set_recall,
        "cluster": cluster,
        "multihop_shape": shape,
    }


def _report(*rows: dict) -> dict:
    return {"per_query": list(rows), "overall": {}, "by_category": {}}


def _multi(query_id: str, cluster: str, shape: str = "two-facet") -> BenchQuery:
    return BenchQuery(
        id=query_id,
        text=f"text for {query_id}",
        category="multi-hop",
        qrels={"bench-doc-a": 2, "bench-doc-b": 1},
        required=("bench-doc-a", "bench-doc-b"),
        cluster=cluster,
        multihop_shape=shape,
    )


def _existing(query_id: str, category: str = "lexical-exact") -> BenchQuery:
    return BenchQuery(
        id=query_id,
        text=f"text for {query_id}",
        category=category,
        qrels={"bench-doc-a": 2},
    )


def _paired_reports(
    queries: list[BenchQuery],
    *,
    gate_ndcg: dict[str, float],
    gate_recall: dict[str, float],
    rrf_ndcg: dict[str, float],
    rrf_recall: dict[str, float],
) -> dict:
    def report(values_ndcg: dict[str, float], values_recall: dict[str, float]) -> dict:
        return {
            "per_query": [
                _row(
                    query.id,
                    query.category,
                    ndcg=values_ndcg[query.id],
                    set_recall=values_recall[query.id],
                    cluster=query.cluster,
                    shape=query.multihop_shape,
                )
                for query in queries
            ]
        }

    return {
        GATE_ARM: report(gate_ndcg, gate_recall),
        "rrf": report(rrf_ndcg, rrf_recall),
    }


class TestScoreArm:
    def test_set_metrics_are_attached_per_query_and_aggregated(self):
        queries = [
            _multi("q1", "c1", "bridge"),
            _existing("q2"),
        ]

        def arm(query: str, top_k: int) -> list[tuple[str, float]]:
            if query == "text for q1":
                return [("bench-doc-a", 2.0), ("bench-doc-b", 1.0)]
            return [("bench-doc-a", 1.0)]

        report = score_arm(arm, queries)
        rows = {row["query_id"]: row for row in report["per_query"]}

        assert rows["q1"]["set_recall@10"] == 1.0
        assert rows["q1"]["full_support@10"] == 1.0
        # The existing categories have no required set: the metric is skipped,
        # not silently 1.0, exactly as recall skips an unjudged query.
        assert rows["q2"]["set_recall@10"] is None
        assert rows["q2"]["full_support@10"] is None
        assert rows["q1"]["cluster"] == "c1"
        assert rows["q1"]["multihop_shape"] == "bridge"
        assert report["overall"]["set_recall@10"] == 1.0
        assert "set_recall@10" in report["by_category"]["multi-hop"]


class TestGateSection:
    def test_a_clear_win_throughout_is_met(self):
        queries = [
            _multi("m1", "c1"),
            _multi("m2", "c1"),
            _multi("m3", "c2"),
            _multi("m4", "c2"),
            _existing("e1"),
            _existing("e2"),
        ]
        reports = _paired_reports(
            queries,
            gate_ndcg={q.id: 0.60 for q in queries},
            gate_recall={q.id: 1.0 for q in queries},
            rrf_ndcg={q.id: 0.50 for q in queries},
            rrf_recall={q.id: 0.50 for q in queries},
        )

        section = gate_section(reports, queries)

        assert section["outcome"] == OUTCOME_MET
        assert section["ndcg@10"]["primary_cluster"]["point"] == pytest.approx(0.10)
        assert section["ndcg@10"]["primary_cluster"]["ci_lower"] > 0.0
        assert section["set_recall@10"]["primary_cluster"]["point"] == 0.50
        assert section["queries"] == 4
        assert section["clusters"] == 2

    def test_a_failed_regression_check_is_not_met(self):
        queries = [_multi("m1", "c1"), _multi("m2", "c2"), _existing("e1"), _existing("e2")]
        reports = _paired_reports(
            queries,
            gate_ndcg={q.id: 0.60 if q.category == "multi-hop" else 0.30 for q in queries},
            gate_recall={q.id: 1.0 for q in queries},
            rrf_ndcg={q.id: 0.55 if q.category == "multi-hop" else 0.60 for q in queries},
            rrf_recall={q.id: 0.5 for q in queries},
        )

        section = gate_section(reports, queries)

        assert section["regression"]["passed"] is False
        assert section["outcome"] == OUTCOME_NOT_MET

    def test_a_flagged_category_makes_a_met_gate_inconclusive(self):
        queries = [
            _multi("m1", "c1"),
            _multi("m2", "c2"),
            _existing("e1", "lexical-exact"),
            _existing("e2", "lexical-exact"),
            _existing("e3", "lexical-exact"),
            _existing("e4", "lexical-exact"),
            _existing("e5", "version-pinning"),
        ]
        rrf = {q.id: 0.50 for q in queries}
        gate = dict(rrf)
        # One flagged category, but an overall delta inside the floor.
        gate["e5"] = 0.44
        gate["m1"] = gate["m2"] = 0.60
        reports = _paired_reports(
            queries,
            gate_ndcg=gate,
            gate_recall={q.id: 1.0 for q in queries},
            rrf_ndcg=rrf,
            rrf_recall={q.id: 0.5 for q in queries},
        )

        section = gate_section(reports, queries)

        assert section["regression"]["passed"] is True
        assert section["regression"]["flagged_categories"] == ["version-pinning"]
        assert section["outcome"] == OUTCOME_INCONCLUSIVE


class TestRegressionSection:
    def test_only_the_existing_categories_enter(self):
        queries = [_multi("m1", "c1"), _multi("m2", "c2"), _existing("e1")]
        reports = _paired_reports(
            queries,
            gate_ndcg={q.id: 0.5 for q in queries},
            gate_recall={q.id: 1.0 for q in queries},
            rrf_ndcg={q.id: 0.5 for q in queries},
            rrf_recall={q.id: 0.5 for q in queries},
        )

        section = regression_section(reports, queries)

        assert section["queries"] == 1
        assert section["overall_delta"] == 0.0
        assert section["passed"] is True
        assert section["flagged_categories"] == []


class TestAggregationAndShapeSections:
    def test_aggregation_is_reported_and_never_gates(self):
        queries = [
            BenchQuery(
                id=f"a{i}",
                text=f"agg {i}",
                category="aggregation",
                qrels={"bench-doc-a": 2, "bench-doc-b": 1},
                required=("bench-doc-a", "bench-doc-b"),
                cluster=f"g{i}",
            )
            for i in range(4)
        ]
        reports = _paired_reports(
            queries,
            gate_ndcg={q.id: 0.6 for q in queries},
            gate_recall={q.id: 1.0 for q in queries},
            rrf_ndcg={q.id: 0.5 for q in queries},
            rrf_recall={q.id: 0.5 for q in queries},
        )

        section = aggregation_section(reports, queries)

        assert section["gating"] is False
        assert section["queries"] == 4
        assert section["clusters"] == 4
        assert section["ndcg@10"]["cluster_bootstrap"]["point"] == pytest.approx(0.1)

    def test_the_shape_breakdown_partitions_the_multi_hop_queries(self):
        queries = [
            _multi("b1", "bridge-1", "bridge"),
            _multi("b2", "bridge-1", "bridge"),
            _multi("b3", "bridge-2", "bridge"),
            _multi("b4", "bridge-2", "bridge"),
            _multi("t1", "two-1"),
            _multi("t2", "two-1"),
            _multi("t3", "two-2"),
            _multi("t4", "two-2"),
        ]
        rrf = {q.id: 0.5 for q in queries}
        gate = dict(rrf)
        for query_id in ("b1", "b2", "b3", "b4"):
            gate[query_id] = 0.7
        reports = _paired_reports(
            queries,
            gate_ndcg=gate,
            gate_recall={q.id: 1.0 for q in queries},
            rrf_ndcg=rrf,
            rrf_recall={q.id: 0.5 for q in queries},
        )

        section = shape_breakdown(reports, queries)

        assert section["bridge"]["queries"] == 4
        assert section["bridge"]["clusters"] == 2
        assert section["bridge"]["ndcg@10"]["cluster_bootstrap"]["point"] == pytest.approx(0.2)
        assert section["two-facet"]["queries"] == 4
        assert section["two-facet"]["ndcg@10"]["cluster_bootstrap"]["point"] == pytest.approx(0.0)
        for shape in ("bridge", "two-facet"):
            assert section[shape]["gating"] is False
            assert section[shape]["threshold_applies"] is False


class TestPerQueryDiagnostics:
    def _gold(self, tmp_path: Path) -> Path:
        path = tmp_path / "gold.json"
        path.write_text(
            json.dumps(
                {
                    "queries": {
                        "q-alpha": {
                            "category": "multi-hop",
                            "documents": ["bench-doc-a", "bench-doc-b"],
                            "entities": ["Alpha"],
                            "joining_entity": "Alpha",
                            "facts": ["Alpha links to Beta"],
                        },
                        "q-missing": {
                            "category": "multi-hop",
                            "documents": ["bench-doc-a", "bench-doc-b"],
                            "entities": ["Nowhere"],
                            "joining_entity": "Nowhere",
                            "facts": ["Nothing holds"],
                        },
                        "q-agg": {
                            "category": "aggregation",
                            "documents": ["bench-doc-a"],
                            "entities": ["Beta"],
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_fidelity_per_query_names_the_connected_and_the_missing(self, tmp_path):
        from docugraph.evals.graph_bench import fidelity_per_query

        fixture = build_synthetic_fixture(tmp_path)
        rows = fidelity_per_query(fixture, self._gold(tmp_path))

        assert set(rows) == {"q-alpha", "q-missing"}
        assert rows["q-alpha"]["joining_present"] is True
        assert rows["q-alpha"]["joins_connected"] is True
        assert rows["q-alpha"]["missing_entities"] == []
        assert rows["q-alpha"]["missing_facts"] == []
        assert rows["q-missing"]["joining_present"] is False
        assert rows["q-missing"]["joins_connected"] is False
        assert rows["q-missing"]["missing_entities"] == ["nowhere"]
        assert rows["q-missing"]["missing_facts"] == ["nothing holds"]

    def test_entity_linking_recall_uses_the_bench_group_only(self, tmp_path):
        from docugraph.evals.graph_bench import entity_linking_recall

        fixture = build_synthetic_fixture(tmp_path)
        queries = [
            BenchQuery(
                id="q-alpha",
                text="alpha",
                category="multi-hop",
                qrels={"bench-doc-a": 2, "bench-doc-b": 1},
                required=("bench-doc-a", "bench-doc-b"),
                cluster="c1",
                multihop_shape="bridge",
            ),
            BenchQuery(
                id="q-missing",
                text="alpha",
                category="multi-hop",
                qrels={"bench-doc-a": 2, "bench-doc-b": 1},
                required=("bench-doc-a", "bench-doc-b"),
                cluster="c2",
                multihop_shape="two-facet",
            ),
        ]

        rows = entity_linking_recall(
            fixture,
            embedder=KeywordEmbedder(),
            queries=queries,
            gold_path=self._gold(tmp_path),
        )

        assert rows["q-alpha"]["linked"] is True
        assert rows["q-alpha"]["seed_rank"] == 1
        # Four bench entities; the fifth (a decoy in group "other") must not
        # occupy a seed slot, exactly as the arm's snapshot scopes the search.
        assert rows["q-alpha"]["seeds"] == 4
        assert rows["q-missing"]["linked"] is False
        assert rows["q-missing"]["joining_present"] is False
