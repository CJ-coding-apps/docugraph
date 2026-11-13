"""Benchmark dataset: loading, validation, and integrity.

The dataset is three committed files under ``benchmarks/retrieval/``:

- ``corpus.jsonl`` -- one document per line: ``{id, title, role, category,
  content}``. One corpus document is one chunk, deliberately: the chunker is
  bypassed so that relevance judgments are pinned to stable ids and cannot
  rot when chunking changes. Chunking quality is a different benchmark.
- ``queries.jsonl`` -- one query per line: ``{id, text, category, qrels,
  design_note}``, plus ``required`` for multi-hop and aggregation queries.
  ``qrels`` maps corpus ids to grades (1 partial, 2 answers the query);
  everything absent is grade 0. ``required`` is the support set a complete
  answer needs (every document that must be joined or aggregated), which is
  what the set metrics score; it is the same set recorded in
  ``gold_subgraph.json``. ``design_note`` records the trap the query sets, so
  a future editor knows what a change would un-measure.
- ``gold_subgraph.json`` -- the support graph for multi-hop and aggregation
  queries: required documents, joining/aggregated entities, required facts.
- ``MANIFEST.json`` -- sha256 of every committed dataset file, counts, and a
  dataset version.
  Metrics are only comparable against the same judgments, so editing the
  dataset is a deliberate act: it must update the manifest, and it
  invalidates committed baselines.

Standard-library only, like metrics.py, and for the same reason.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Every query and corpus doc carries one of these; the report aggregates per
# category, so an unknown category is a dataset bug, not a new feature.
CATEGORIES = (
    "lexical-exact",
    "paraphrase",
    "keyword-trap",
    "semantic-distractor",
    "version-pinning",
    "acronym-synonym",
    "negation-constraint",
    # Support-set categories: the answer is not one document. The graph bench
    # scores them with set recall / full support over an explicit required set.
    "multi-hop",
    "aggregation",
)

# Categories whose answer needs more than one document, and therefore must
# declare ``required``. A query in one of these that leaves the support set
# implicit cannot be scored by the set metrics, only silently skipped.
SUPPORT_SET_CATEGORIES = ("multi-hop", "aggregation")

ROLES = ("target", "distractor", "background")


@dataclass(frozen=True)
class CorpusDoc:
    id: str
    title: str
    role: str  # target | distractor | background -- analysis only, never ranked on
    category: str | None  # the trap this doc serves; None for background
    content: str

    @property
    def text(self) -> str:
        """What gets indexed: title folded into the body, as real docs have it.

        The vector store embeds and FTS-indexes the ``content`` column only,
        so a title kept solely in metadata would be invisible to retrieval.
        """
        return f"# {self.title}\n\n{self.content}"


@dataclass(frozen=True)
class BenchQuery:
    id: str
    text: str
    category: str
    qrels: dict[str, int] = field(default_factory=dict)
    design_note: str = ""
    # None means the field is absent; an empty tuple means it was present but
    # empty, which is a dataset bug the validator reports rather than erasing.
    required: tuple[str, ...] | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}:{line_no}: not valid JSON: {e}") from e
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_corpus(dataset_dir: Path) -> list[CorpusDoc]:
    rows = _read_jsonl(dataset_dir / "corpus.jsonl")
    return [
        CorpusDoc(
            id=r["id"],
            title=r["title"],
            role=r["role"],
            category=r.get("category"),
            content=r["content"],
        )
        for r in rows
    ]


def load_queries(dataset_dir: Path) -> list[BenchQuery]:
    rows = _read_jsonl(dataset_dir / "queries.jsonl")
    return [
        BenchQuery(
            id=r["id"],
            text=r["text"],
            category=r["category"],
            qrels={k: int(v) for k, v in r["qrels"].items()},
            design_note=r.get("design_note", ""),
            required=(None if "required" not in r else tuple(r["required"])),
        )
        for r in rows
    ]


def load_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(
        (dataset_dir / "MANIFEST.json").read_text(encoding="utf-8")
    )
    return manifest


def validate(dataset_dir: Path) -> list[str]:
    """Every way the dataset can lie, checked. Returns problems, empty = sound.

    Run as a unit test and before every bench run: a benchmark over a dataset
    whose judgments don't resolve measures nothing, quietly.
    """
    problems: list[str] = []

    corpus = load_corpus(dataset_dir)
    queries = load_queries(dataset_dir)
    manifest = load_manifest(dataset_dir)

    # Integrity: every file the manifest names is what the manifest says it
    # is. Files are added to the map as they are committed (gold_subgraph.json,
    # and later the staged graph fixture), so the loop follows the manifest
    # rather than a hardcoded list.
    for name, expected in manifest["sha256"].items():
        path = dataset_dir / name
        if not path.exists():
            problems.append(f"{name}: named by the manifest but missing")
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append(
                f"{name}: sha256 {actual} != manifest {expected} -- the dataset "
                "changed without the manifest; update MANIFEST.json and regenerate baselines"
            )

    if manifest["counts"]["corpus"] != len(corpus):
        problems.append(
            f"manifest says {manifest['counts']['corpus']} corpus docs, found {len(corpus)}"
        )
    if manifest["counts"]["queries"] != len(queries):
        problems.append(
            f"manifest says {manifest['counts']['queries']} queries, found {len(queries)}"
        )

    # Ids are unique and qrels resolve.
    corpus_ids = [d.id for d in corpus]
    dup = {i for i in corpus_ids if corpus_ids.count(i) > 1}
    if dup:
        problems.append(f"duplicate corpus ids: {sorted(dup)}")
    id_set = set(corpus_ids)

    query_ids = [q.id for q in queries]
    dupq = {i for i in query_ids if query_ids.count(i) > 1}
    if dupq:
        problems.append(f"duplicate query ids: {sorted(dupq)}")

    for q in queries:
        for doc_id, grade in q.qrels.items():
            if doc_id not in id_set:
                problems.append(f"{q.id}: qrel '{doc_id}' names no corpus doc")
            if grade not in (1, 2):
                problems.append(f"{q.id}: grade {grade} for '{doc_id}' (judged grades are 1 or 2)")
        if not any(g == 2 for g in q.qrels.values()):
            problems.append(f"{q.id}: no grade-2 document -- every query must be answerable")
        if q.category not in CATEGORIES:
            problems.append(f"{q.id}: unknown category '{q.category}'")
        if not q.design_note:
            problems.append(f"{q.id}: missing design_note -- the trap must be recorded")

        # The support set: explicit, resolvable, judged relevant, and big
        # enough to be a set. A required document at grade 0 would mean the
        # set metrics demand something the judgments call irrelevant.
        if q.required is not None:
            if not q.required:
                problems.append(
                    f"{q.id}: required is present but empty -- omit the field "
                    "or list the support set"
                )
            for doc_id in q.required:
                if doc_id not in id_set:
                    problems.append(f"{q.id}: required '{doc_id}' names no corpus doc")
                elif q.qrels.get(doc_id, 0) < 1:
                    problems.append(
                        f"{q.id}: required '{doc_id}' is not in qrels at grade >= 1"
                    )
        if q.category in SUPPORT_SET_CATEGORIES:
            if q.required is None:
                problems.append(
                    f"{q.id}: {q.category} query has no required support set"
                )
            elif len(set(q.required)) < 2:
                problems.append(
                    f"{q.id}: {q.category} query needs at least two required documents"
                )

    for d in corpus:
        if d.role not in ROLES:
            problems.append(f"{d.id}: unknown role '{d.role}'")
        if d.role in ("target", "distractor") and d.category not in CATEGORIES:
            problems.append(
                f"{d.id}: role '{d.role}' requires a known category, got '{d.category}'"
            )

    # Every category is actually exercised.
    covered = {q.category for q in queries}
    missing = [c for c in CATEGORIES if c not in covered]
    if missing:
        problems.append(f"categories with no queries: {missing}")

    return problems


def default_dataset_dir() -> Path:
    """The committed dataset, resolved from the repository root.

    parents[3] of this file is the checkout root (evals -> docugraph -> src ->
    root). The dataset ships in the repository, not in the wheel: the benchmark
    is a development instrument, and a `pip install docugraph` user has no
    committed corpus to be scored against. The CLI checks existence and says
    so rather than guessing.
    """
    return Path(__file__).resolve().parents[3] / "benchmarks" / "retrieval"
