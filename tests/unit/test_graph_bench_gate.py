"""The pre-registered gate and the fidelity row (WP §5, §4.3; G1-G3, P1).

The numbers asserted here come from
``docs/graph-bench-preregistration-2026-11-13.md``. They are not tunable test
parameters: changing one invalidates the registration, so each is asserted
against its literal value.

G1-G3 use synthetic per-query deltas, not arms, so this module proves the
statistics and the outcome mapping on their own.
"""

import json

import pytest

from docugraph.evals import graph_bench as gb

from .graph_bench_helpers import build_synthetic_fixture


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return build_synthetic_fixture(tmp_path_factory.mktemp("graph-bench-gate"))


class TestPreregisteredNumbers:
    def test_fixed_numbers_are_the_registered_ones(self):
        """Every number the pre-registration fixes, in one place."""
        assert gb.GRAPH_GROUP_ID == "bench"
        assert gb.SEED_CAP == 20
        assert gb.EDGE_CAP == 50
        assert gb.RRF_K == 60
        assert gb.BOOTSTRAP_RESAMPLES == 10_000
        assert gb.BOOTSTRAP_SEED == 20261113
        assert gb.NDCG_THRESHOLD == 0.05
        assert gb.SET_RECALL_THRESHOLD == 0.10
        assert gb.REGRESSION_FLOOR == -0.02
        assert gb.CATEGORY_FLAG_FLOOR == -0.05
        assert gb.GATE_ARM == "rrf+graph-2hop"
        assert gb.FALLBACK_GATE_ARM == "rrf+graph"

    def test_outcome_strings_are_the_registered_ones(self):
        assert (gb.OUTCOME_MET, gb.OUTCOME_NOT_MET, gb.OUTCOME_INCONCLUSIVE) == (
            "MET",
            "NOT MET",
            "INCONCLUSIVE",
        )


class TestBootstrap:
    def test_fixed_seed_reproduces_the_interval(self):
        """G1: same queries, same seed, same bounds -- exactly."""
        deltas = [-0.1, 0.0, 0.05, 0.2, 0.3, 0.15, -0.02, 0.4,
                  -0.2, 0.1, 0.25, 0.0, 0.05, 0.12, 0.3, 0.08]
        first = gb.metric_result(deltas, threshold=gb.NDCG_THRESHOLD)
        second = gb.metric_result(deltas, threshold=gb.NDCG_THRESHOLD)
        assert (
            first.point,
            first.ci_lower,
            first.ci_upper,
        ) == (
            second.point,
            second.ci_lower,
            second.ci_upper,
        ), "the bootstrap is not reproducible with a fixed seed"
        assert first.ci_lower <= first.point <= first.ci_upper

    def test_point_estimate_is_the_mean_delta(self):
        result = gb.metric_result([0.2, 0.4], threshold=gb.NDCG_THRESHOLD)
        assert result.point == pytest.approx(0.3)


class TestClusterBootstrap:
    """D-8: the resampling unit is the evidence cluster, not the query.

    These deltas are deliberately correlated *within* a cluster: both queries of
    each pair carry the same delta, exactly the pattern the multi-hop queries
    have (two questions about one document pair). Treating them as independent
    understates the interval; resampling the cluster keeps the pair together.
    """

    DELTAS = [-0.10, -0.10, -0.05, -0.05, 0.00, 0.00, 0.05, 0.05,
              0.10, 0.10, 0.15, 0.15, 0.20, 0.20, 0.25, 0.25]
    CLUSTERS = [f"cluster-{index // 2}" for index in range(16)]

    def _clustered(self) -> gb.MetricResult:
        return gb.metric_result(
            self.DELTAS, threshold=gb.NDCG_THRESHOLD, clusters=self.CLUSTERS
        )

    def test_clustered_interval_is_reproducible(self):
        """G1 for the primary interval: same clusters, same seed, same bounds."""
        first = self._clustered()
        second = self._clustered()
        assert (first.ci_lower, first.ci_upper) == (second.ci_lower, second.ci_upper)
        assert first.ci_lower <= first.point <= first.ci_upper

    def test_point_estimate_stays_the_mean_over_queries(self):
        clustered = self._clustered()
        per_query = gb.metric_result(self.DELTAS, threshold=gb.NDCG_THRESHOLD)
        assert clustered.point == per_query.point

    def test_mutation_resampling_by_query_instead_of_by_cluster_turns_it_red(self):
        """The mutation check the pre-registration names.

        If ``clusters`` is ignored -- the interval resampled by query -- the
        interval is narrower on correlated deltas. Requiring a strictly wider
        clustered interval makes that mutation fail here, not in a report.
        """
        clustered = self._clustered()
        per_query = gb.metric_result(self.DELTAS, threshold=gb.NDCG_THRESHOLD)

        clustered_width = clustered.ci_upper - clustered.ci_lower
        per_query_width = per_query.ci_upper - per_query.ci_lower
        assert clustered_width > per_query_width, (
            "the clustered interval is not wider than the per-query interval on "
            "within-cluster correlated deltas: the resampling unit is ignored"
        )
        assert (clustered.ci_lower, clustered.ci_upper) != (
            per_query.ci_lower,
            per_query.ci_upper,
        )

    def test_one_cluster_id_per_query_is_required(self):
        with pytest.raises(ValueError, match="one id per query"):
            gb.metric_result(
                self.DELTAS, threshold=gb.NDCG_THRESHOLD, clusters=self.CLUSTERS[:-1]
            )


class TestGateOnSyntheticArms:
    def test_better_arm_is_met_and_noise_is_not(self):
        """G2: a clear effect passes; noise does not."""
        better = [0.2] * 16
        noise = [0.3 if index % 2 else -0.3 for index in range(16)]

        better_outcome = gb.evaluate_gate(
            ndcg=gb.metric_result(better, threshold=gb.NDCG_THRESHOLD),
            set_recall=gb.metric_result(better, threshold=gb.SET_RECALL_THRESHOLD),
            regression_ok=True,
        )
        assert better_outcome == gb.OUTCOME_MET

        noise_outcome = gb.evaluate_gate(
            ndcg=gb.metric_result(noise, threshold=gb.NDCG_THRESHOLD),
            set_recall=gb.metric_result(noise, threshold=gb.SET_RECALL_THRESHOLD),
            regression_ok=True,
        )
        assert noise_outcome in {gb.OUTCOME_NOT_MET, gb.OUTCOME_INCONCLUSIVE}


class TestOutcomeMapping:
    def _met(self, threshold: float) -> gb.MetricResult:
        return gb.MetricResult(
            point=threshold + 0.15, ci_lower=0.05, ci_upper=0.4, threshold=threshold
        )

    def test_a_ci_above_zero_and_point_above_threshold_is_met(self):
        """G3(a)."""
        ndcg = self._met(gb.NDCG_THRESHOLD)
        recall = self._met(gb.SET_RECALL_THRESHOLD)
        assert gb.metric_outcome(ndcg) == gb.OUTCOME_MET
        assert (
            gb.evaluate_gate(ndcg=ndcg, set_recall=recall, regression_ok=True)
            == gb.OUTCOME_MET
        )

    def test_b_ci_upper_bound_below_threshold_is_not_met(self):
        """G3(b)."""
        ndcg = gb.MetricResult(
            point=0.01, ci_lower=-0.05, ci_upper=0.03, threshold=gb.NDCG_THRESHOLD
        )
        recall = self._met(gb.SET_RECALL_THRESHOLD)
        assert gb.metric_outcome(ndcg) == gb.OUTCOME_NOT_MET
        assert (
            gb.evaluate_gate(ndcg=ndcg, set_recall=recall, regression_ok=True)
            == gb.OUTCOME_NOT_MET
        )

    def test_c_ci_spanning_zero_with_point_above_threshold_is_inconclusive(self):
        """G3(c): the point estimate clears the bar, the interval does not."""
        ndcg = gb.MetricResult(
            point=0.2, ci_lower=-0.1, ci_upper=0.4, threshold=gb.NDCG_THRESHOLD
        )
        recall = self._met(gb.SET_RECALL_THRESHOLD)
        assert gb.metric_outcome(ndcg) == gb.OUTCOME_INCONCLUSIVE
        assert (
            gb.evaluate_gate(ndcg=ndcg, set_recall=recall, regression_ok=True)
            == gb.OUTCOME_INCONCLUSIVE
        )

    def test_d_one_hop_fallback_records_a_negative_result_as_inconclusive(self):
        """G3(d): a 1-hop arm is not a fair test of multi-hop."""
        ndcg = gb.MetricResult(
            point=0.01, ci_lower=-0.05, ci_upper=0.03, threshold=gb.NDCG_THRESHOLD
        )
        recall = self._met(gb.SET_RECALL_THRESHOLD)
        assert (
            gb.evaluate_gate(
                ndcg=ndcg,
                set_recall=recall,
                regression_ok=True,
                gate_arm=gb.FALLBACK_GATE_ARM,
            )
            == gb.OUTCOME_INCONCLUSIVE
        )

    def test_regression_failure_is_not_met(self):
        assert (
            gb.evaluate_gate(
                ndcg=self._met(gb.NDCG_THRESHOLD),
                set_recall=self._met(gb.SET_RECALL_THRESHOLD),
                regression_ok=False,
            )
            == gb.OUTCOME_NOT_MET
        )


class TestFidelityRow:
    def test_missing_gold_entity_is_reported_and_the_gate_is_unaffected(
        self, synthetic, tmp_path
    ):
        """P1: the fidelity row diagnoses the extractor; it never gates.

        Gold names are lowercase on purpose: the row normalises names, so the
        comparison must be case-insensitive. ``zeta`` is absent from the
        fixture and must be named as missing.
        """
        gold = {
            "queries": {
                "mh-1": {
                    "category": "multi-hop",
                    "documents": ["bench-doc-a", "bench-doc-b"],
                    "entities": ["alpha", "beta", "zeta"],
                    "joining_entity": "alpha",
                    "facts": ["alpha links to beta", "gamma links to delta"],
                }
            }
        }
        gold_path = tmp_path / "gold_subgraph.json"
        gold_path.write_text(json.dumps(gold), encoding="utf-8")

        row = gb.fidelity_row(synthetic, gold_path)

        assert row["entities_present"] == pytest.approx(2 / 3)
        assert "zeta" in [name.casefold() for name in row["missing_entities"]]
        assert row["facts_present"] == pytest.approx(1.0)
        assert row["joins_connected"] == pytest.approx(1.0)
        assert row["gating"] is False, "the fidelity row is declared non-gating"

        # The gate ignores the row entirely: full fidelity is not an input.
        met = gb.MetricResult(
            point=0.2, ci_lower=0.1, ci_upper=0.4, threshold=gb.NDCG_THRESHOLD
        )
        met_recall = gb.MetricResult(
            point=0.2, ci_lower=0.1, ci_upper=0.4, threshold=gb.SET_RECALL_THRESHOLD
        )
        assert (
            gb.evaluate_gate(ndcg=met, set_recall=met_recall, regression_ok=True)
            == gb.OUTCOME_MET
        )
