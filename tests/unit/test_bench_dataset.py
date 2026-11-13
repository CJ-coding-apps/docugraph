"""The committed benchmark dataset is sound, and the validator can tell when one isn't.

The first half runs the validator over ``benchmarks/retrieval`` -- the dataset
the numbers in the README come from. The second half feeds the validator
broken datasets and expects it to object: a validator that cannot fail is a
comment, not a gate (the same reasoning as the live-reranker workflow header).
"""

import json
from pathlib import Path

import pytest

from docugraph.evals.dataset import (
    CATEGORIES,
    SUPPORT_SET_CATEGORIES,
    default_dataset_dir,
    load_corpus,
    load_queries,
    sha256_file,
    validate,
)

DATASET = default_dataset_dir()


class TestCommittedDataset:
    def test_dataset_exists_where_the_cli_looks(self):
        assert (DATASET / "MANIFEST.json").exists(), (
            f"no dataset at {DATASET}; default_dataset_dir and the committed "
            "files have drifted apart"
        )

    def test_validates_clean(self):
        assert validate(DATASET) == []

    def test_every_category_has_a_target_answer(self):
        queries = load_queries(DATASET)
        for category in CATEGORIES:
            in_category = [q for q in queries if q.category == category]
            assert in_category, f"no queries for {category}"
            for q in in_category:
                assert any(g == 2 for g in q.qrels.values())

    def test_support_set_queries_declare_a_judged_support_set(self):
        """Multi-hop and aggregation answers are sets; the set must be explicit."""
        for q in load_queries(DATASET):
            if q.category not in SUPPORT_SET_CATEGORIES:
                continue
            assert q.required is not None, f"{q.id} has no required set"
            assert len(set(q.required)) >= 2, f"{q.id} needs a real support set"
            for doc_id in q.required:
                assert q.qrels.get(doc_id, 0) >= 1, (
                    f"{q.id}: required {doc_id} is not judged relevant"
                )

    def test_cluster_shape_is_the_preregistered_one(self):
        """D-8 fixes 16 multi-hop clusters (32 queries) and 4 aggregation groups.

        Also the structural assumptions the bootstrap relies on: one support
        set per cluster, and no document shared between clusters of a category.
        """
        queries = load_queries(DATASET)
        by_category = {
            category: [q for q in queries if q.category == category]
            for category in SUPPORT_SET_CATEGORIES
        }
        multi_hop = by_category["multi-hop"]
        aggregation = by_category["aggregation"]
        assert len(multi_hop) == 32, f"{len(multi_hop)} multi-hop queries"
        assert len({q.cluster for q in multi_hop}) == 16, "multi-hop clusters"
        assert len(aggregation) == 16, f"{len(aggregation)} aggregation queries"
        assert len({q.cluster for q in aggregation}) == 4, "aggregation clusters"

        for category, in_category in by_category.items():
            by_cluster: dict[str, list] = {}
            for q in in_category:
                by_cluster.setdefault(q.cluster, []).append(q)
            documents_seen: dict[str, str] = {}
            for cluster, members in by_cluster.items():
                assert len(members) > 1, f"{cluster} is a singleton, not a cluster"
                support_sets = {frozenset(q.required or ()) for q in members}
                assert len(support_sets) == 1, f"{cluster} has more than one support set"
                for doc_id in next(iter(support_sets)):
                    other = documents_seen.setdefault(doc_id, cluster)
                    assert other == cluster, (
                        f"{category}: {doc_id} is in {other} and {cluster}"
                    )

    def test_gold_subgraph_covers_exactly_the_support_set_queries(self):
        """The gold subgraph and the queries must name the same queries/docs.

        The fidelity row reads the gold file directly, so a drift between the
        two would make extraction coverage describe a different benchmark.
        """
        gold_path = DATASET / "gold_subgraph.json"
        assert gold_path.exists(), "gold_subgraph.json is missing"
        gold = json.loads(gold_path.read_text(encoding="utf-8"))["queries"]
        support = {
            q.id: q for q in load_queries(DATASET)
            if q.category in SUPPORT_SET_CATEGORIES
        }
        assert set(gold) == set(support)
        for query_id, record in gold.items():
            q = support[query_id]
            assert record["category"] == q.category
            assert set(record["documents"]) == set(q.required or ())
            if q.category == "multi-hop":
                assert record.get("joining_entity"), (
                    f"{query_id}: multi-hop gold record needs a joining_entity"
                )
            else:
                assert record.get("entities"), (
                    f"{query_id}: aggregation gold record needs aggregated entities"
                )
            assert record.get("facts"), f"{query_id}: gold record needs required facts"

    def test_distractors_are_never_graded_relevant(self):
        """A distractor graded relevant for its own category is a broken trap."""
        corpus = {d.id: d for d in load_corpus(DATASET)}
        for q in load_queries(DATASET):
            for doc_id, grade in q.qrels.items():
                doc = corpus[doc_id]
                if doc.role == "distractor" and doc.category == q.category:
                    # The one sanctioned exception: a lexical-exact index page
                    # may earn partial credit for containing the exact token
                    # with a one-liner (see q-lex-1's design note). Grade 2
                    # would still be a broken trap.
                    assert grade == 1, (
                        f"{q.id}: distractor {doc_id} graded {grade} in its own category"
                    )


class TestRequiredField:
    def test_absent_and_present_are_distinguishable(self, tmp_path):
        corpus = [GOOD_DOC, {**GOOD_DOC, "id": "d2"}]
        queries = [
            GOOD_QUERY,
            {**GOOD_QUERY, "id": "q2", "qrels": {"d1": 2, "d2": 2}, "required": ["d1", "d2"]},
        ]
        d = _write_dataset(tmp_path, corpus, queries)
        loaded = {q.id: q for q in load_queries(d)}
        assert loaded["q1"].required is None
        assert loaded["q2"].required == ("d1", "d2")


class TestClusterField:
    """D-8: support-set queries carry the id of their evidence cluster."""

    def test_cluster_is_loaded_and_absent_is_none(self, tmp_path):
        corpus = [GOOD_DOC, {**GOOD_DOC, "id": "d2"}]
        queries = [
            GOOD_QUERY,
            {
                **GOOD_QUERY,
                "id": "q2",
                "category": "multi-hop",
                "qrels": {"d1": 2, "d2": 1},
                "required": ["d1", "d2"],
                "cluster": "d1+d2",
            },
        ]
        d = _write_dataset(tmp_path, corpus, queries)
        loaded = {q.id: q for q in load_queries(d)}
        assert loaded["q1"].cluster is None
        assert loaded["q2"].cluster == "d1+d2"


def _write_dataset(tmp_path: Path, corpus: list[dict], queries: list[dict]) -> Path:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    corpus_path.write_text("".join(json.dumps(d) + "\n" for d in corpus), encoding="utf-8")
    queries_path.write_text("".join(json.dumps(q) + "\n" for q in queries), encoding="utf-8")
    manifest = {
        "dataset": "test",
        "version": "0",
        "counts": {"corpus": len(corpus), "queries": len(queries)},
        "sha256": {
            "corpus.jsonl": sha256_file(corpus_path),
            "queries.jsonl": sha256_file(queries_path),
        },
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


GOOD_DOC = {"id": "d1", "title": "T", "role": "target", "category": "paraphrase", "content": "x"}
GOOD_QUERY = {
    "id": "q1",
    "text": "t",
    "category": "paraphrase",
    "qrels": {"d1": 2},
    "design_note": "n",
}


class TestValidatorCatches:
    def test_minimal_sound_dataset_passes_everything_but_coverage(self, tmp_path):
        d = _write_dataset(tmp_path, [GOOD_DOC], [GOOD_QUERY])
        problems = validate(d)
        # One query can't cover every category; nothing else may be wrong.
        assert len(problems) == 1
        assert "categories with no queries" in problems[0]

    def test_tampered_file_fails_integrity(self, tmp_path):
        d = _write_dataset(tmp_path, [GOOD_DOC], [GOOD_QUERY])
        with (d / "corpus.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({**GOOD_DOC, "id": "d2"}) + "\n")
        assert any("sha256" in p for p in validate(d))

    def test_dangling_qrel_is_caught(self, tmp_path):
        bad_query = {**GOOD_QUERY, "qrels": {"nope": 2}}
        d = _write_dataset(tmp_path, [GOOD_DOC], [bad_query])
        assert any("names no corpus doc" in p for p in validate(d))

    def test_unanswerable_query_is_caught(self, tmp_path):
        bad_query = {**GOOD_QUERY, "qrels": {"d1": 1}}
        d = _write_dataset(tmp_path, [GOOD_DOC], [bad_query])
        assert any("no grade-2" in p for p in validate(d))

    def test_missing_design_note_is_caught(self, tmp_path):
        bad_query = {**GOOD_QUERY, "design_note": ""}
        d = _write_dataset(tmp_path, [GOOD_DOC], [bad_query])
        assert any("design_note" in p for p in validate(d))

    def test_unknown_category_is_caught(self, tmp_path):
        bad_query = {**GOOD_QUERY, "category": "vibes"}
        d = _write_dataset(tmp_path, [GOOD_DOC], [bad_query])
        assert any("unknown category" in p for p in validate(d))

    def test_invalid_grade_is_caught(self, tmp_path):
        bad_query = {**GOOD_QUERY, "qrels": {"d1": 3}}
        d = _write_dataset(tmp_path, [GOOD_DOC], [bad_query])
        problems = validate(d)
        assert any("grade 3" in p for p in problems)

    def test_required_document_must_resolve_and_be_judged(self, tmp_path):
        dangling = {**GOOD_QUERY, "required": ["d1", "nope"]}
        d = _write_dataset(tmp_path, [GOOD_DOC], [dangling])
        assert any("required 'nope' names no corpus doc" in p for p in validate(d))

        unjudged_doc = {**GOOD_DOC, "id": "d2"}
        unjudged = {**GOOD_QUERY, "required": ["d1", "d2"]}
        d = _write_dataset(tmp_path, [GOOD_DOC, unjudged_doc], [unjudged])
        assert any(
            "required 'd2' is not in qrels at grade >= 1" in p for p in validate(d)
        )

    def test_present_but_empty_required_is_caught(self, tmp_path):
        bad_query = {**GOOD_QUERY, "required": []}
        d = _write_dataset(tmp_path, [GOOD_DOC], [bad_query])
        assert any("required is present but empty" in p for p in validate(d))

    def test_support_set_category_without_required_is_caught(self, tmp_path):
        bad_query = {**GOOD_QUERY, "category": "multi-hop"}
        d = _write_dataset(tmp_path, [GOOD_DOC], [bad_query])
        assert any("has no required support set" in p for p in validate(d))

    def test_a_one_document_support_set_is_caught(self, tmp_path):
        bad_query = {**GOOD_QUERY, "category": "aggregation", "required": ["d1"]}
        d = _write_dataset(tmp_path, [GOOD_DOC], [bad_query])
        assert any("at least two required documents" in p for p in validate(d))

    def test_a_support_set_query_without_a_cluster_is_caught(self, tmp_path):
        """D-8: without a cluster id every query is its own resampling unit."""
        bad_query = {
            **GOOD_QUERY,
            "category": "multi-hop",
            "qrels": {"d1": 2, "d2": 1},
            "required": ["d1", "d2"],
        }
        d = _write_dataset(tmp_path, [GOOD_DOC, {**GOOD_DOC, "id": "d2"}], [bad_query])
        assert any("has no cluster" in p for p in validate(d))

    def test_a_cluster_on_a_non_support_set_query_is_caught(self, tmp_path):
        """A cluster id there would imply a resampling unit that does not exist."""
        bad_query = {**GOOD_QUERY, "cluster": "d1+d1"}
        d = _write_dataset(tmp_path, [GOOD_DOC], [bad_query])
        assert any("cluster" in p and "non-support-set" in p for p in validate(d))

    def test_one_cluster_id_with_two_support_sets_is_caught(self, tmp_path):
        """A cluster is one evidence unit: its queries must ask the same pair."""
        corpus = [GOOD_DOC, {**GOOD_DOC, "id": "d2"}, {**GOOD_DOC, "id": "d3"}]
        first = {
            **GOOD_QUERY,
            "id": "q1",
            "category": "multi-hop",
            "qrels": {"d1": 2, "d2": 1},
            "required": ["d1", "d2"],
            "cluster": "shared",
        }
        second = {
            **first,
            "id": "q2",
            "qrels": {"d2": 2, "d3": 1},
            "required": ["d2", "d3"],
        }
        d = _write_dataset(tmp_path, corpus, [first, second])
        assert any("different support sets" in p for p in validate(d))

    def test_a_document_in_two_clusters_of_one_category_is_caught(self, tmp_path):
        """Shared documents correlate the units the bootstrap calls independent."""
        corpus = [GOOD_DOC, {**GOOD_DOC, "id": "d2"}, {**GOOD_DOC, "id": "d3"}]
        first = {
            **GOOD_QUERY,
            "id": "q1",
            "category": "multi-hop",
            "qrels": {"d1": 2, "d2": 1},
            "required": ["d1", "d2"],
            "cluster": "c1",
        }
        second = {
            **first,
            "id": "q2",
            "qrels": {"d2": 2, "d3": 1},
            "required": ["d2", "d3"],
            "cluster": "c2",
        }
        d = _write_dataset(tmp_path, corpus, [first, second])
        assert any("not independent" in p for p in validate(d))

    def test_a_cluster_id_shared_across_categories_is_caught(self, tmp_path):
        """One cluster id is one kind of evidence unit, never two."""
        corpus = [GOOD_DOC, {**GOOD_DOC, "id": "d2"}]
        multi_hop = {
            **GOOD_QUERY,
            "id": "q1",
            "category": "multi-hop",
            "qrels": {"d1": 2, "d2": 1},
            "required": ["d1", "d2"],
            "cluster": "shared",
        }
        aggregation = {**multi_hop, "id": "q2", "category": "aggregation"}
        d = _write_dataset(tmp_path, corpus, [multi_hop, aggregation])
        assert any("used by both" in p for p in validate(d))

    def test_a_non_string_cluster_id_is_caught(self, tmp_path):
        """A JSON number would be coerced by a naive loader."""
        bad_query = {
            **GOOD_QUERY,
            "category": "multi-hop",
            "qrels": {"d1": 2, "d2": 1},
            "required": ["d1", "d2"],
            "cluster": 5,
        }
        d = _write_dataset(tmp_path, [GOOD_DOC, {**GOOD_DOC, "id": "d2"}], [bad_query])
        assert any("invalid cluster id" in p for p in validate(d))

    def test_a_whitespace_cluster_id_is_caught(self, tmp_path):
        bad_query = {
            **GOOD_QUERY,
            "category": "multi-hop",
            "qrels": {"d1": 2, "d2": 1},
            "required": ["d1", "d2"],
            "cluster": "   ",
        }
        d = _write_dataset(tmp_path, [GOOD_DOC, {**GOOD_DOC, "id": "d2"}], [bad_query])
        assert any("has no cluster id" in p for p in validate(d))

    def test_manifest_naming_a_missing_file_is_caught(self, tmp_path):
        d = _write_dataset(tmp_path, [GOOD_DOC], [GOOD_QUERY])
        manifest = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
        manifest["sha256"]["gold_subgraph.json"] = "0" * 64
        (d / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        assert any("gold_subgraph.json: named by the manifest" in p for p in validate(d))

    def test_malformed_jsonl_names_file_and_line(self, tmp_path):
        d = _write_dataset(tmp_path, [GOOD_DOC], [GOOD_QUERY])
        with (d / "queries.jsonl").open("a", encoding="utf-8") as f:
            f.write("{not json\n")
        with pytest.raises(ValueError, match=r"queries\.jsonl:2"):
            validate(d)
