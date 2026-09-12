"""Unit tests for the reranker: fastembed cross-encoder + factory default.

Uses an injected fake TextCrossEncoder so no ONNX weights are downloaded.
The one live-inference test is gated on the ~1.8 GB model already being
cached, so it never triggers a download in CI.
"""

import os
from pathlib import Path

import pytest

from docugraph.core.models import Chunk, SearchResult
from docugraph.retrieval.reranker import (
    FastembedReranker,
    PassthroughReranker,
    RerankerType,
    get_reranker,
)


def _v2m3_cached() -> bool:
    """True if the bge-reranker-v2-m3 ONNX weights are already on disk."""
    cache = Path(os.environ.get("FASTEMBED_CACHE_PATH", Path.home() / ".cache" / "fastembed"))
    return cache.exists() and any("bge-reranker-v2-m3" in p.name.lower() for p in cache.glob("*"))


class FakeCrossEncoder:
    """Stand-in for fastembed's TextCrossEncoder.

    Returns the score list it is configured with, in document order.
    """

    def __init__(self, scores):
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, documents):
        docs = list(documents)
        self.calls.append((query, docs))
        return self._scores[: len(docs)]


def _result(content: str, score: float) -> SearchResult:
    return SearchResult(chunk=Chunk(document_id="d", content=content), score=score)


class TestFastembedReranker:
    def test_reranks_by_cross_encoder_score(self):
        # Initial order a,b,c by retrieval score; cross-encoder flips it.
        results = [
            _result("a", 0.9),
            _result("b", 0.5),
            _result("c", 0.1),
        ]
        # Give "c" the highest cross-encoder score → it should rank first.
        model = FakeCrossEncoder(scores=[0.0, 0.2, 1.0])
        reranker = FastembedReranker(model_name="fake", score_weight=1.0, model=model)

        reranked = reranker.rerank("q", results)

        assert [r.chunk.content for r in reranked] == ["c", "b", "a"]
        # Cross-encoder saw the query and all document contents.
        assert model.calls[0][0] == "q"
        assert model.calls[0][1] == ["a", "b", "c"]

    def test_empty_results(self):
        reranker = FastembedReranker(model_name="fake", model=FakeCrossEncoder([]))
        assert reranker.rerank("q", []) == []

    def test_top_k_slices(self):
        results = [_result(c, 0.5) for c in ("a", "b", "c", "d")]
        model = FakeCrossEncoder(scores=[0.1, 0.2, 0.3, 0.4])
        reranker = FastembedReranker(model_name="fake", model=model)

        reranked = reranker.rerank("q", results, top_k=2)
        assert len(reranked) == 2
        # Highest cross-encoder scores were d then c.
        assert [r.chunk.content for r in reranked] == ["d", "c"]

    def test_score_weight_blends_original_and_rerank(self):
        results = [_result("a", 1.0), _result("b", 0.0)]
        # Cross-encoder prefers b, but weight=0 means only original score counts.
        model = FakeCrossEncoder(scores=[0.0, 1.0])
        reranker = FastembedReranker(model_name="fake", score_weight=0.0, model=model)

        reranked = reranker.rerank("q", results)
        # With weight 0, original order (a first) is preserved.
        assert reranked[0].chunk.content == "a"


class TestRegisterV2M3:
    def test_registration_is_idempotent(self):
        """add_custom_model raising ValueError (already registered) is swallowed."""
        calls = {"n": 0}

        class Encoder:
            @staticmethod
            def add_custom_model(**_kwargs):
                calls["n"] += 1
                raise ValueError("already registered")

        # Should not raise despite the ValueError.
        FastembedReranker._register_v2m3(Encoder)
        FastembedReranker._register_v2m3(Encoder)
        assert calls["n"] == 2

    def test_registration_passes_a_real_modelsource(self):
        """sources must be a fastembed ModelSource, not a dict.

        fastembed's download path reads `model.sources.hf`, so a plain dict
        registers fine but crashes at fetch time — regression guard.
        """
        from fastembed.common.model_description import ModelSource

        captured = {}

        class Encoder:
            @staticmethod
            def add_custom_model(*, sources, **_kwargs):
                captured["sources"] = sources

        FastembedReranker._register_v2m3(Encoder)
        assert isinstance(captured["sources"], ModelSource)
        assert captured["sources"].hf == FastembedReranker._V2M3_HF_REPO


@pytest.mark.skipif(
    not _v2m3_cached(),
    reason="bge-reranker-v2-m3 (~1.8 GB) not cached; skipping live-inference test",
)
class TestFastembedRerankerLive:
    def test_real_model_reorders_and_scores(self):
        """End-to-end with the real ONNX cross-encoder (no fakes).

        The clearly on-topic passage must rank first with a higher blended
        score than an unrelated one — proves registration, download wiring
        (ModelSource), and inference all work together.
        """
        results = [
            SearchResult(
                chunk=Chunk(document_id="d", content="Bananas are a yellow tropical fruit."),
                score=0.5,
            ),
            SearchResult(
                chunk=Chunk(
                    document_id="d",
                    content=(
                        "Reranking reorders search results with a cross-encoder "
                        "to improve precision."
                    ),
                ),
                score=0.5,
            ),
        ]
        reranker = FastembedReranker()  # default bge-reranker-v2-m3
        reranked = reranker.rerank("how does reranking improve precision", results)

        assert "Reranking" in reranked[0].chunk.content
        assert reranked[0].final_score > reranked[1].final_score


class TestGetReranker:
    def test_default_is_passthrough(self):
        reranker = get_reranker()
        assert isinstance(reranker, PassthroughReranker)
        assert reranker.name == "none"

    def test_passthrough_preserves_order(self):
        results = [_result("a", 0.3), _result("b", 0.9)]
        reranked = get_reranker(RerankerType.NONE).rerank("q", results)
        assert [r.chunk.content for r in reranked] == ["a", "b"]
