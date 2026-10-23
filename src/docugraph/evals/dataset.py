"""Benchmark dataset: loading, validation, and integrity.

The dataset is three committed files under ``benchmarks/retrieval/``:

- ``corpus.jsonl`` -- one document per line: ``{id, title, role, category,
  content}``. One corpus document is one chunk, deliberately: the chunker is
  bypassed so that relevance judgments are pinned to stable ids and cannot
  rot when chunking changes. Chunking quality is a different benchmark.
- ``queries.jsonl`` -- one query per line: ``{id, text, category, qrels,
  design_note}``. ``qrels`` maps corpus ids to grades (1 partial, 2 answers
  the query); everything absent is grade 0. ``design_note`` records the trap
  the query sets, so a future editor knows what a change would un-measure.
- ``MANIFEST.json`` -- sha256 of both files, counts, and a dataset version.
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
)

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


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
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
        )
        for r in rows
    ]


def load_manifest(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "MANIFEST.json").read_text(encoding="utf-8"))


def validate(dataset_dir: Path) -> list[str]:
    """Every way the dataset can lie, checked. Returns problems, empty = sound.

    Run as a unit test and before every bench run: a benchmark over a dataset
    whose judgments don't resolve measures nothing, quietly.
    """
    problems: list[str] = []

    corpus = load_corpus(dataset_dir)
    queries = load_queries(dataset_dir)
    manifest = load_manifest(dataset_dir)

    # Integrity: the files are what the manifest says they are.
    for name in ("corpus.jsonl", "queries.jsonl"):
        actual = sha256_file(dataset_dir / name)
        expected = manifest["sha256"][name]
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
