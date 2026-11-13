"""The pre-registered gate and the fidelity row (WP §5, §4.3; G1-G3, P1).

The numbers asserted here come from
``docs/graph-bench-preregistration-2026-11-13.md``. They are not tunable test
parameters: changing one invalidates the registration, so each is asserted
against its literal value.

G1-G3 use synthetic per-query deltas, not arms, so this module proves the
statistics and the outcome mapping on their own.
"""

import json
import random
from types import SimpleNamespace

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

    Each cluster's two deltas differ slightly but move together, the pattern
    the multi-hop queries have (two questions about one document pair). The
    small within-cluster difference is deliberate: with identical members, an
    implementation that lets only the first member of each drawn cluster enter
    is indistinguishable from the correct one.
    """

    DELTAS = [-0.10, -0.06, -0.05, -0.01, 0.00, 0.04, 0.05, 0.09,
              0.10, 0.14, 0.15, 0.19, 0.20, 0.24, 0.25, 0.29]
    CLUSTERS = [f"cluster-{index // 2}" for index in range(16)]

    # Pinned bounds. They pin the output; the seed and the resample count are
    # pinned separately (several wrong seeds and counts reproduce these bounds
    # on a fixture this small), and the percentile by its contract test.
    # Recompute only with a change to the pre-registration.
    GOLDEN_CLUSTERED = (0.01375, 0.17625)
    GOLDEN_PER_QUERY = (0.03875, 0.15125)

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

    def test_registered_bounds_are_pinned(self):
        """G1: the registered seed, resample count and percentile, not just
        reproducibility -- a wrong seed is perfectly reproducible."""
        clustered = self._clustered()
        assert (clustered.ci_lower, clustered.ci_upper) == self.GOLDEN_CLUSTERED
        per_query = gb.metric_result(self.DELTAS, threshold=gb.NDCG_THRESHOLD)
        assert (per_query.ci_lower, per_query.ci_upper) == self.GOLDEN_PER_QUERY

    def test_the_registered_seed_and_resample_count_are_used(self, monkeypatch):
        """G1: record what ``metric_result`` asks the generator for.

        The golden bounds alone cannot pin these: on a fixture this small,
        several wrong seeds and resample counts reproduce them exactly. A spy
        on the generator closes that hole.
        """
        created = []

        class Recording:
            def __init__(self, seed):
                self.seed = seed
                self.calls = 0
                self._inner = random.Random(seed)

            def randrange(self, stop):
                self.calls += 1
                return self._inner.randrange(stop)

        def factory(seed):
            instance = Recording(seed)
            created.append(instance)
            return instance

        monkeypatch.setattr(gb, "random", SimpleNamespace(Random=factory))
        clustered = self._clustered()
        per_query = gb.metric_result(self.DELTAS, threshold=gb.NDCG_THRESHOLD)

        assert [instance.seed for instance in created] == [
            gb.BOOTSTRAP_SEED,
            gb.BOOTSTRAP_SEED,
        ], "the registered seed is not what the bootstrap is constructed with"
        # One draw per cluster per resample, then one draw per query per resample.
        assert created[0].calls == gb.BOOTSTRAP_RESAMPLES * 8
        assert created[1].calls == gb.BOOTSTRAP_RESAMPLES * len(self.DELTAS)
        # The spy delegates to the real generator, so the bounds are unchanged.
        assert (clustered.ci_lower, clustered.ci_upper) == self.GOLDEN_CLUSTERED
        assert (per_query.ci_lower, per_query.ci_upper) == self.GOLDEN_PER_QUERY

    def test_percentiles_are_nearest_rank(self):
        """The registered method: rank ceil(q*n), 1-based, not interpolated."""
        values = [float(index) for index in range(10_000)]
        assert gb._percentile(values, 0.025) == 249.0
        assert gb._percentile(values, 0.975) == 9749.0

    def test_a_single_cluster_is_refused(self):
        """One cluster resamples to itself: a zero-width, false-certainty CI."""
        with pytest.raises(ValueError, match="at least two distinct clusters"):
            gb.metric_result(
                self.DELTAS, threshold=gb.NDCG_THRESHOLD, clusters=["only"] * 16
            )

    def test_cluster_ids_must_be_non_empty_strings(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            gb.metric_result(
                self.DELTAS, threshold=gb.NDCG_THRESHOLD, clusters=[None] * 16
            )

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

    def test_clustered_arms_map_the_same_way(self):
        """G2 (D-8): the outcome mapping is the same under cluster resampling."""
        clusters = [f"pair-{index // 2}" for index in range(16)]
        # Two queries per pair; pairs vary only mildly, so the effect is real.
        better = [0.15 + 0.1 * (index // 8) for index in range(16)]
        noise = [0.3 if (index // 2) % 2 == 0 else -0.3 for index in range(16)]

        better_outcome = gb.evaluate_gate(
            ndcg=gb.metric_result(
                better, threshold=gb.NDCG_THRESHOLD, clusters=clusters
            ),
            set_recall=gb.metric_result(
                better, threshold=gb.SET_RECALL_THRESHOLD, clusters=clusters
            ),
            regression_ok=True,
        )
        assert better_outcome == gb.OUTCOME_MET

        noise_outcome = gb.evaluate_gate(
            ndcg=gb.metric_result(
                noise, threshold=gb.NDCG_THRESHOLD, clusters=clusters
            ),
            set_recall=gb.metric_result(
                noise, threshold=gb.SET_RECALL_THRESHOLD, clusters=clusters
            ),
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

    def test_e_fully_bootstrapped_outcomes_match_the_mapping(self):
        """G3 (D-8): (a)-(d) end to end through the cluster bootstrap.

        The earlier tests feed hand-written intervals; these feed deltas and
        let the registered machinery produce the intervals, so the mapping is
        proven against the real statistic, not a stand-in.
        """
        clusters = [f"pair-{index // 2}" for index in range(16)]

        def outcome(deltas, *, gate_arm=gb.GATE_ARM):
            return gb.evaluate_gate(
                ndcg=gb.metric_result(
                    deltas, threshold=gb.NDCG_THRESHOLD, clusters=clusters
                ),
                set_recall=gb.metric_result(
                    deltas, threshold=gb.SET_RECALL_THRESHOLD, clusters=clusters
                ),
                regression_ok=True,
                gate_arm=gate_arm,
            )

        # (a) every pair clearly positive.
        assert outcome([0.19, 0.29] * 8) == gb.OUTCOME_MET
        # (b) every pair near zero: the effect size is ruled out.
        assert outcome([0.00, 0.02] * 8) == gb.OUTCOME_NOT_MET
        # (c) pairs disagree in sign: the interval spans zero.
        disagreement = [0.6 if (index // 2) % 2 == 0 else -0.4 for index in range(16)]
        assert outcome(disagreement) == gb.OUTCOME_INCONCLUSIVE
        # (d) the same (b) under the 1-hop fallback is not a fair test.
        assert outcome([0.00, 0.02] * 8, gate_arm=gb.FALLBACK_GATE_ARM) == (
            gb.OUTCOME_INCONCLUSIVE
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
