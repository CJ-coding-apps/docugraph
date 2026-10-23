"""Benchmark runner: ingest the corpus into a fresh store, run arms, score.

Every run owns its store: a temporary LanceDB directory that exists for the
run and is deleted after it. Scoring against whatever happens to be in a
user's index would make every number a function of their crawl history.

The embedder is injected. The real benchmark runs on the package default
(fastembed, bge-small-en-v1.5) so numbers describe what ships; the integration
test injects a deterministic fake so the harness is proven without model
weights.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docugraph.evals.arms import Arm, build_arms
from docugraph.evals.dataset import (
    BenchQuery,
    CorpusDoc,
    load_corpus,
    load_manifest,
    load_queries,
    validate,
)
from docugraph.evals.metrics import (
    macro_average,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    stable_ranking,
)

REPORT_SCHEMA = "docugraph-bench/1"
KS = (1, 5, 10)
TOP_K = 10


def ingest_corpus(vector_store: Any, corpus: list[CorpusDoc]) -> int:
    """Load the corpus into the store, one doc = one chunk, stable ids."""
    from docugraph.core.models import Chunk

    chunks = [
        Chunk(
            id=doc.id,
            document_id=doc.id,
            content=doc.text,
            metadata={"title": doc.title, "role": doc.role, "category": doc.category or ""},
        )
        for doc in corpus
    ]
    added: int = vector_store.add_chunks(chunks)
    # Build the FTS index now rather than on first search, so arm timings and
    # failures land where they belong.
    vector_store.ensure_fts_index()
    return added


def _score_query(ranked: list[str], query: BenchQuery) -> dict[str, Any]:
    row: dict[str, Any] = {
        "query_id": query.id,
        "category": query.category,
        "top": ranked[:TOP_K],
    }
    for k in KS:
        row[f"recall@{k}"] = recall_at_k(ranked, query.qrels, k)
    row[f"mrr@{TOP_K}"] = mrr_at_k(ranked, query.qrels, TOP_K)
    row[f"ndcg@{TOP_K}"] = ndcg_at_k(ranked, query.qrels, TOP_K)
    first = next(
        (i for i, doc_id in enumerate(ranked, 1) if query.qrels.get(doc_id, 0) >= 1),
        None,
    )
    row["first_relevant_rank"] = first
    return row


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [f"recall@{k}" for k in KS] + [f"mrr@{TOP_K}", f"ndcg@{TOP_K}"]
    agg: dict[str, Any] = {"n_queries": len(rows)}
    for name in metric_names:
        agg[name] = macro_average([r[name] for r in rows])
    return agg


def run_arm(arm: Arm, queries: list[BenchQuery]) -> dict[str, Any]:
    per_query = []
    for q in queries:
        ranked = stable_ranking(arm(q.text, TOP_K))
        per_query.append(_score_query(ranked, q))

    by_category: dict[str, Any] = {}
    for category in sorted({q.category for q in queries}):
        by_category[category] = _aggregate([r for r in per_query if r["category"] == category])

    return {
        "overall": _aggregate(per_query),
        "by_category": by_category,
        "per_query": per_query,
    }


def _environment(vector_store: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    """What produced these numbers. A report that omits this can't be compared."""
    from importlib import metadata as im

    from docugraph._version import __version__

    def version_of(package: str) -> str | None:
        try:
            return im.version(package)
        except Exception:
            return None

    return {
        "embedder": vector_store.embedding_model,
        "docugraph": __version__,
        "lancedb": version_of("lancedb"),
        "fastembed": version_of("fastembed"),
        "dataset_version": manifest["version"],
        "corpus_sha256": manifest["sha256"]["corpus.jsonl"],
        "queries_sha256": manifest["sha256"]["queries.jsonl"],
    }


def run_bench(
    dataset_dir: Path,
    store_dir: Path,
    *,
    embedder: Any | None = None,
    arm_names: list[str] | None = None,
    include_rerank: bool = False,
) -> dict[str, Any]:
    """Run the benchmark end to end and return the report dict.

    Raises ValueError when the dataset fails validation: numbers over a
    dataset whose judgments don't resolve measure nothing.
    """
    problems = validate(dataset_dir)
    if problems:
        raise ValueError("dataset is not sound:\n" + "\n".join(f"- {p}" for p in problems))

    corpus = load_corpus(dataset_dir)
    queries = load_queries(dataset_dir)
    manifest = load_manifest(dataset_dir)

    from docugraph.storage.vector_store import VectorStore

    vector_store = VectorStore(db_path=store_dir, embedder=embedder)
    ingest_corpus(vector_store, corpus)

    arms = build_arms(vector_store, include_rerank=include_rerank)
    if arm_names:
        unknown = [a for a in arm_names if a not in arms]
        if unknown:
            raise ValueError(f"unknown arms: {unknown}; available: {sorted(arms)}")
        arms = {name: arms[name] for name in arm_names}

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "top_k": TOP_K,
        "environment": _environment(vector_store, manifest),
        "arms": {},
    }
    for name, arm in arms.items():
        report["arms"][name] = run_arm(arm, queries)
    return report
