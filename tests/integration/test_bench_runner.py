"""The benchmark harness, proven end to end without model weights.

A deterministic hashed bag-of-tokens embedder stands in for fastembed, the
same injection seam the real run uses in reverse -- so these tests prove the
plumbing (ingest with stable ids, FTS index, every arm, scoring, report
shape) while downloading nothing. The *numbers* these tests produce are
meaningless by construction; nothing here asserts retrieval quality. Quality
numbers come from `docugraph bench run` on the real embedder, and live in the
committed baseline.
"""

import hashlib
import math

import pytest

from docugraph.core.embeddings import EmbeddingProviderBase
from docugraph.evals.dataset import default_dataset_dir, load_queries
from docugraph.evals.runner import REPORT_SCHEMA, run_bench

DATASET = default_dataset_dir()
DIMS = 64


class HashedBagEmbedder(EmbeddingProviderBase):
    """Deterministic token-hash embeddings: same text, same vector, any machine."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._one(query)

    @property
    def dimensions(self) -> int:
        return DIMS

    @property
    def model_name(self) -> str:
        return "test-hashed-bag-64"

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * DIMS
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            vec[digest[0] % DIMS] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    """One full run over the committed dataset; module-scoped because the
    arms and queries all read from the same ingested store."""
    store_dir = tmp_path_factory.mktemp("bench-store")
    return run_bench(
        DATASET,
        store_dir,
        embedder=HashedBagEmbedder(),
        arm_names=["vector", "fts", "rrf", "weighted"],
    )


class TestReportShape:
    def test_schema_and_environment(self, report):
        assert report["schema"] == REPORT_SCHEMA
        env = report["environment"]
        assert env["embedder"] == "test-hashed-bag-64"
        # A report must say which judgments produced it.
        assert len(env["corpus_sha256"]) == 64
        assert len(env["queries_sha256"]) == 64

    def test_every_arm_scored_every_query(self, report):
        n_queries = len(load_queries(DATASET))
        for arm_name, arm_report in report["arms"].items():
            assert arm_report["overall"]["n_queries"] == n_queries, arm_name
            assert len(arm_report["per_query"]) == n_queries, arm_name

    def test_metrics_are_probabilities(self, report):
        for arm_report in report["arms"].values():
            for row in arm_report["per_query"]:
                for name in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"):
                    value = row[name]
                    assert value is None or 0.0 <= value <= 1.0

    def test_per_category_aggregates_cover_every_query_category(self, report):
        categories = {q.category for q in load_queries(DATASET)}
        for arm_report in report["arms"].values():
            assert set(arm_report["by_category"]) == categories


class TestArmsActuallyDiffer:
    def test_fts_finds_the_exact_token(self, report):
        """BM25 over a corpus where exactly two docs contain 'QRV-4012' must
        put one of them first -- otherwise the FTS leg isn't FTS (the v0.1.1
        failure this benchmark was built to expose)."""
        per_query = {r["query_id"]: r for r in report["arms"]["fts"]["per_query"]}
        top = per_query["q-lex-1"]["top"]
        assert top, "FTS returned nothing for an exact-token query"
        assert top[0] in ("qrv-err-4012", "qrv-err-index")

    def test_vector_and_fts_rankings_are_not_identical(self, report):
        """If every query ranks identically across the two legs, one of them
        is the other wearing a label -- the original silent-fallback bug."""
        vector_tops = [r["top"] for r in report["arms"]["vector"]["per_query"]]
        fts_tops = [r["top"] for r in report["arms"]["fts"]["per_query"]]
        assert vector_tops != fts_tops


class TestNativeHybridArm:
    def test_lancedb_hybrid_runs_without_fallback(self, tmp_path):
        """Native hybrid must run as hybrid against the pinned LanceDB.

        This is the arm that regressed silently before: the store now records
        its fallback and the arm refuses to score under it, so if this test
        fails, the pinned LanceDB's hybrid API needs attention -- do not relax
        the test to let the arm fall back quietly.
        """
        report = run_bench(
            DATASET,
            tmp_path,
            embedder=HashedBagEmbedder(),
            arm_names=["lancedb-hybrid"],
        )
        overall = report["arms"]["lancedb-hybrid"]["overall"]
        assert overall["n_queries"] == len(load_queries(DATASET))
