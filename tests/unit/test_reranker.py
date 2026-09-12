"""Unit tests for the reranker: fastembed cross-encoder + factory default.

Uses an injected fake TextCrossEncoder so no ONNX weights are downloaded.
"""

from docugraph.core.models import Chunk, SearchResult
from docugraph.retrieval.reranker import (
    FastembedReranker,
    PassthroughReranker,
    RerankerType,
    get_reranker,
)


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


class TestGetReranker:
    def test_default_is_passthrough(self):
        reranker = get_reranker()
        assert isinstance(reranker, PassthroughReranker)
        assert reranker.name == "none"

    def test_passthrough_preserves_order(self):
        results = [_result("a", 0.3), _result("b", 0.9)]
        reranked = get_reranker(RerankerType.NONE).rerank("q", results)
        assert [r.chunk.content for r in reranked] == ["a", "b"]
