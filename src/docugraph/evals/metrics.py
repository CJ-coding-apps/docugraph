"""Retrieval-quality metrics: recall@k, MRR@k, nDCG@k.

Standard-library only, deliberately: these functions are the part of the
benchmark whose correctness everything else rests on, so they must be testable
on a machine with nothing installed, and they must not change behavior when a
dependency does.

Definitions are declared here rather than defaulted, because every one of
these metrics has variants and a number without its definition cannot be
compared:

- Relevance is graded: 0 (irrelevant or unjudged), 1 (partial), 2 (answers
  the query). ``min_grade`` decides what counts as relevant for the binary
  metrics (recall, MRR); the default is 1, so a partial document counts.
- ``recall_at_k`` is |relevant retrieved in top-k| / |relevant in qrels|.
- ``mrr_at_k`` is 1/rank of the first relevant result within the top-k,
  0.0 when none appears.
- ``ndcg_at_k`` uses **linear gain** (gain = grade) with the standard
  log2(rank+1) discount, and the ideal from the query's own sorted grades.
  With grades capped at 2, linear vs exponential gain barely moves the
  numbers, but the choice has to be stated for the numbers to mean anything.
- Queries with no relevant document in qrels return ``None`` (skip), not 0.0:
  a query that cannot be answered must not drag an average down.

Ranking ties are the caller's problem by design: ``stable_ranking`` breaks
score ties by chunk id so that two runs of the same system produce the same
ordering, and therefore the same metrics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def stable_ranking(scored: Sequence[tuple[str, float]]) -> list[str]:
    """Order (chunk_id, score) pairs by score descending, ties by id ascending.

    Metrics are computed over an ordering, so the ordering must be a function
    of the result set alone -- not of dict insertion order or a backend's
    internal tie behavior, either of which would make metric diffs noise.
    """
    return [doc_id for doc_id, _ in sorted(scored, key=lambda p: (-p[1], p[0]))]


def recall_at_k(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
    min_grade: int = 1,
) -> float | None:
    """Fraction of the query's relevant documents found in the top k.

    Returns None when qrels holds no document at or above ``min_grade``:
    there is nothing to recall, and 0.0 would punish the system for the
    dataset's gap.
    """
    relevant = {doc_id for doc_id, grade in qrels.items() if grade >= min_grade}
    if not relevant:
        return None
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def mrr_at_k(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
    min_grade: int = 1,
) -> float | None:
    """Reciprocal rank of the first relevant result in the top k; 0.0 if none.

    Returns None (skip) when the query has no relevant document at all, for
    the same reason as recall_at_k.
    """
    relevant = {doc_id for doc_id, grade in qrels.items() if grade >= min_grade}
    if not relevant:
        return None
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Discounted cumulative gain: sum of grade / log2(rank + 1) over top k."""
    total = 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        grade = qrels.get(doc_id, 0)
        if grade > 0:
            total += grade / math.log2(rank + 1)
    return total


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float | None:
    """Normalized DCG with linear gain; None when the query has no graded doc.

    The ideal ordering is the query's own grades sorted descending -- the best
    any system could do on this query -- so 1.0 means "could not have ranked
    these better within the top k".
    """
    ideal_grades = sorted((g for g in qrels.values() if g > 0), reverse=True)
    if not ideal_grades:
        return None
    ideal = 0.0
    for rank, grade in enumerate(ideal_grades[:k], start=1):
        ideal += grade / math.log2(rank + 1)
    if ideal == 0.0:
        return None
    return dcg_at_k(ranked, qrels, k) / ideal


def macro_average(values: Sequence[float | None]) -> float | None:
    """Mean over the non-None values; None when every query was skipped.

    Macro (per-query mean), not micro: each query is one unit of difficulty,
    and a query with many relevant documents must not outvote one with a
    single hard target.
    """
    kept = [v for v in values if v is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)
