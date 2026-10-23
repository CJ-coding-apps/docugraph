"""Baseline comparison: the regression gate.

``compare_reports`` holds a new report against a committed baseline and
returns every finding. The gate trips on nDCG@10 -- overall or in any single
category -- dropping by more than the tolerance, for any arm the baseline
measured. Per-category matters more than overall here: a change that wins
broadly but destroys one failure mode (say, a reranker that fixes paraphrase
and breaks exact-code lookup) must not pass on its average.

Reports over different judgments are not comparable at all, so a dataset
hash mismatch is a refusal, not a finding. Standard-library only.
"""

from __future__ import annotations

from typing import Any

DEFAULT_TOLERANCE = 0.02
_GATED_METRIC = "ndcg@10"


class IncomparableReports(ValueError):
    """The two reports do not measure the same thing."""


def _check_comparable(baseline: dict[str, Any], new: dict[str, Any]) -> None:
    for key in ("corpus_sha256", "queries_sha256"):
        b = baseline["environment"][key]
        n = new["environment"][key]
        if b != n:
            raise IncomparableReports(
                f"{key} differs (baseline {b[:12]}…, new {n[:12]}…): the dataset "
                "changed, so these numbers answer different questions. "
                "Regenerate the baseline instead of comparing across it."
            )
    if baseline["environment"]["embedder"] != new["environment"]["embedder"]:
        raise IncomparableReports(
            f"embedder differs (baseline {baseline['environment']['embedder']}, "
            f"new {new['environment']['embedder']}): baselines are per-model."
        )


def compare_reports(
    baseline: dict[str, Any],
    new: dict[str, Any],
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[bool, list[dict[str, Any]]]:
    """Compare and return (passed, findings).

    Every finding is a row: ``{arm, scope, metric, baseline, new, delta,
    regression}``. Rows are returned for improvements too -- the gate reads
    ``regression``, people read the whole table.
    """
    _check_comparable(baseline, new)

    findings: list[dict[str, Any]] = []
    passed = True

    for arm_name, base_arm in baseline["arms"].items():
        new_arm = new["arms"].get(arm_name)
        if new_arm is None:
            findings.append(
                {
                    "arm": arm_name,
                    "scope": "overall",
                    "metric": _GATED_METRIC,
                    "baseline": base_arm["overall"].get(_GATED_METRIC),
                    "new": None,
                    "delta": None,
                    "regression": True,
                    "note": "arm measured in the baseline is missing from the new report",
                }
            )
            passed = False
            continue

        scopes: list[tuple[str, dict, dict]] = [
            ("overall", base_arm["overall"], new_arm["overall"])
        ]
        for category, base_cat in base_arm["by_category"].items():
            new_cat = new_arm["by_category"].get(category)
            if new_cat is not None:
                scopes.append((category, base_cat, new_cat))

        for scope, base_scope, new_scope in scopes:
            b = base_scope.get(_GATED_METRIC)
            n = new_scope.get(_GATED_METRIC)
            if b is None or n is None:
                continue
            delta = n - b
            regression = delta < -tolerance
            if regression:
                passed = False
            findings.append(
                {
                    "arm": arm_name,
                    "scope": scope,
                    "metric": _GATED_METRIC,
                    "baseline": round(b, 4),
                    "new": round(n, 4),
                    "delta": round(delta, 4),
                    "regression": regression,
                }
            )

    return passed, findings
