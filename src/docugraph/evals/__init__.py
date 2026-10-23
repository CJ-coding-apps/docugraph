"""Retrieval-quality benchmark ("retrieval goldens").

A committed adversarial corpus and graded judgments
(``benchmarks/retrieval/``), metrics with declared definitions
(``metrics.py``), arms over the real retrieval stack (``arms.py``), a runner
that owns its store (``runner.py``), and a regression gate (``compare.py``).
Entry points: ``docugraph bench run`` and ``docugraph bench compare``.

Deliberately not imported here: ``runner`` and ``arms`` pull in the storage
stack, while ``metrics``, ``dataset`` and ``compare`` are standard-library
only and must stay importable without it.
"""
