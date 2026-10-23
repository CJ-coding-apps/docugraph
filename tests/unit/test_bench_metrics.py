"""Unit tests for the benchmark metrics, against hand-computed values.

Every expected number here was computed by hand from the definitions in
``docugraph.evals.metrics``'s module docstring. If a test fails after an edit
to metrics.py, the definitions changed -- and every committed baseline is
invalidated with them.
"""

import math

from docugraph.evals.metrics import (
    dcg_at_k,
    macro_average,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    stable_ranking,
)

QRELS = {"a": 2, "b": 1}  # everything else grade 0


class TestStableRanking:
    def test_orders_by_score_descending(self):
        assert stable_ranking([("x", 0.1), ("y", 0.9), ("z", 0.5)]) == ["y", "z", "x"]

    def test_ties_break_by_id_so_runs_reproduce(self):
        # Equal scores: ordering must be a function of the result set alone.
        assert stable_ranking([("b", 0.5), ("a", 0.5), ("c", 0.5)]) == ["a", "b", "c"]


class TestRecall:
    def test_all_relevant_found(self):
        assert recall_at_k(["a", "b", "x"], QRELS, 3) == 1.0

    def test_half_found(self):
        assert recall_at_k(["a", "x", "y"], QRELS, 3) == 0.5

    def test_cutoff_applies(self):
        # "b" sits at rank 4, outside k=3.
        assert recall_at_k(["a", "x", "y", "b"], QRELS, 3) == 0.5

    def test_min_grade_2_counts_only_full_answers(self):
        # Only "a" is grade 2; finding "b" alone recalls none of them.
        assert recall_at_k(["b", "x", "y"], QRELS, 3, min_grade=2) == 0.0

    def test_no_relevant_docs_skips_not_zero(self):
        assert recall_at_k(["x"], {}, 3) is None


class TestMRR:
    def test_first_hit_at_rank_one(self):
        assert mrr_at_k(["a", "x"], QRELS, 10) == 1.0

    def test_first_hit_at_rank_three(self):
        assert mrr_at_k(["x", "y", "b"], QRELS, 10) == 1.0 / 3

    def test_no_hit_in_k_is_zero(self):
        assert mrr_at_k(["x", "y"], QRELS, 10) == 0.0

    def test_no_relevant_docs_skips(self):
        assert mrr_at_k(["x"], {}, 10) is None


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        # Ideal order: a (2), b (1).
        assert ndcg_at_k(["a", "b"], QRELS, 10) == 1.0

    def test_swapped_ranking_hand_computed(self):
        # DCG([b, a]) = 1/log2(2) + 2/log2(3); IDCG = 2/log2(2) + 1/log2(3).
        dcg = 1 / math.log2(2) + 2 / math.log2(3)
        idcg = 2 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(["b", "a"], QRELS, 10) == dcg / idcg

    def test_relevant_outside_k_scores_zero(self):
        assert ndcg_at_k(["x", "y"], QRELS, 2) == 0.0

    def test_dcg_ignores_unjudged(self):
        assert dcg_at_k(["x", "y", "z"], QRELS, 3) == 0.0

    def test_no_graded_docs_skips(self):
        assert ndcg_at_k(["x"], {}, 10) is None


class TestMacroAverage:
    def test_mean_over_values(self):
        assert macro_average([1.0, 0.0, 0.5]) == 0.5

    def test_none_values_are_skipped_not_zeroed(self):
        # A skipped query must not drag the average down.
        assert macro_average([1.0, None]) == 1.0

    def test_all_skipped_is_none(self):
        assert macro_average([None, None]) is None
