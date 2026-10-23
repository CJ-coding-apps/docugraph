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
        # One query can't cover seven categories; nothing else may be wrong.
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

    def test_malformed_jsonl_names_file_and_line(self, tmp_path):
        d = _write_dataset(tmp_path, [GOOD_DOC], [GOOD_QUERY])
        with (d / "queries.jsonl").open("a", encoding="utf-8") as f:
            f.write("{not json\n")
        with pytest.raises(ValueError, match=r"queries\.jsonl:2"):
            validate(d)
