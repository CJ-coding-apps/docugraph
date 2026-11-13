"""The set-level metrics the graph bench adds (WP §4.2, worked examples M1/M2).

``set_recall@k`` and ``full_support@k`` are new, so the numbers they produce
must be anchored to something already trusted: on the existing seven
categories, ``set_recall`` has to equal today's ``recall`` on the committed
baseline. If it does not, every existing report is silently redefined.

M1 uses the committed 0.4.0 baseline's ranked lists rather than a fresh run:
the point is that the *definition* agrees with the one those numbers came from,
and it makes the test cheap and offline.
"""

import json

from docugraph.evals.dataset import default_dataset_dir, load_queries
from docugraph.evals.metrics import (
    full_support_at_k,
    recall_at_k,
    set_recall_at_k,
)

DATASET = default_dataset_dir()
BASELINE = DATASET / "baselines" / "bge-small-en-v1.5.json"
KS = (1, 5, 10)


def _relevant(qrels: dict[str, int]) -> set[str]:
    """Today's relevant set: a partial (grade 1) document counts."""
    return {doc_id for doc_id, grade in qrels.items() if grade >= 1}


class TestSetRecallEqualsExistingRecall:
    def test_set_recall_equals_recall_on_all_56_queries(self):
        """M1: on the existing categories the new metric is the old one.

        Every arm, every committed query, every reported cutoff. A mismatch
        names the arm, query, and cutoff -- a bare assert would hide which
        definition drifted.
        """
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        queries = {query.id: query for query in load_queries(DATASET)}
        assert len(queries) == 56, f"{DATASET}/queries.jsonl has {len(queries)} queries, expected 56"

        for arm_name, arm in baseline["arms"].items():
            for row in arm["per_query"]:
                query = queries[row["query_id"]]
                required = _relevant(query.qrels)
                for k in KS:
                    actual = set_recall_at_k(row["top"], required, k)
                    expected = recall_at_k(row["top"], query.qrels, k)
                    assert actual == expected, (
                        f"{arm_name}/{query.id}: set_recall@{k}={actual} != "
                        f"recall@{k}={expected} (required={sorted(required)}, top={row['top']})"
                    )

    def test_a_missing_required_chunk_is_a_partial_recall(self):
        ranked = ["x", "a", "y"]
        required = {"a", "b", "c"}
        assert set_recall_at_k(ranked, required, 10) == 1 / 3
        assert set_recall_at_k(ranked, required, 1) == 0.0


class TestFullSupport:
    def test_one_of_two_required_chunks_found(self):
        """M2: one of two required chunks found is 0.0 full support, 0.5 recall."""
        ranked = ["a", "x", "y"]
        required = {"a", "b"}
        assert full_support_at_k(ranked, required, 10) == 0.0
        assert set_recall_at_k(ranked, required, 10) == 0.5

    def test_full_support_needs_the_whole_set_inside_the_cutoff(self):
        assert full_support_at_k(["a", "b"], {"a", "b"}, 10) == 1.0
        assert full_support_at_k(["a", "b"], {"a", "b"}, 1) == 0.0


    def test_empty_required_set_is_skipped_not_vacuously_full(self):
        """No support set means nothing to measure: None, like recall's skip.

        A vacuous 1.0 would let a query with no judgments inflate the average.
        """
        assert full_support_at_k(["a"], set(), 10) is None
        assert set_recall_at_k(["a"], set(), 10) is None
