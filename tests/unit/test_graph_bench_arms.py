"""The graph arms: no LLM at query time, refusal on an embedder mismatch,
namespace isolation, and the episode-to-chunk mapping (D2, D3, N1, R-4).

Every test here runs on a synthetic fixture in ``tmp_path``: a small P0-5
graph, a raising LLM client, and a deterministic 4-dimensional embedder. No
model download, no network, no API key.
"""

import pytest

from docugraph.evals import graph_bench as gb

from .graph_bench_helpers import (
    ALL_GRAPH_ARMS,
    SYNTH_DIM,
    SYNTH_DOC_IDS,
    SYNTH_EMBEDDER_MODEL,
    KeywordEmbedder,
    build_synthetic_fixture,
    make_raising_llm,
    run_fixture_arms,
    synthetic_records,
)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return build_synthetic_fixture(tmp_path_factory.mktemp("graph-bench-arms"))


class TestNoLLMInTheQueryPath:
    def test_no_graph_arm_touches_the_llm_client(self, synthetic, tmp_path):
        """D2: replace the LLM with one that raises; run every graph arm."""
        llm = make_raising_llm()
        results = run_fixture_arms(
            synthetic,
            ["alpha", "gamma"],
            db_path=tmp_path / "db-d2",
            embedder=KeywordEmbedder(),
            doc_ids=SYNTH_DOC_IDS,
            llm_client=llm,
        )

        assert set(results) == ALL_GRAPH_ARMS, (
            f"expected all graph arms {sorted(ALL_GRAPH_ARMS)}, got {sorted(results)}"
        )
        for arm_name, rows in results.items():
            assert len(rows) == 2, f"{arm_name} did not run both queries"
        assert llm.calls == 0, (
            f"the query path called the LLM {llm.calls} time(s); "
            "graphiti search must never generate"
        )


class TestEmbedderRefusal:
    def _assert_all_refused(self, manifest, embedder):
        statuses = gb.graph_arm_statuses(
            fixture_manifest=manifest, embedder=embedder
        )
        assert set(statuses) == ALL_GRAPH_ARMS
        for arm_name, status in statuses.items():
            assert status.available is False, f"{arm_name} accepted a bad embedder"
            reason = status.reason or ""
            assert SYNTH_EMBEDDER_MODEL in reason, (
                f"{arm_name}: reason does not name the fixture embedder: {reason!r}"
            )
            assert embedder.model_name in reason, (
                f"{arm_name}: reason does not name the selected embedder: {reason!r}"
            )

        with pytest.raises(gb.GraphArmUnavailable) as excinfo:
            gb.build_graph_arms(
                store=None,
                embedder=embedder,
                fixture_manifest=manifest,
                base_arms={},
                doc_ids=SYNTH_DOC_IDS,
            )
        assert embedder.model_name in str(excinfo.value), (
            "the refusal does not name the offending embedder"
        )

    def test_cloud_embedder_is_refused_with_a_status(self, synthetic):
        """D3: a cloud embedder (different id and width) is never silently used."""
        manifest = gb.load_fixture_manifest(synthetic)
        cloud = KeywordEmbedder(model_name="cloud/some-embedder", dimensions=1536)
        self._assert_all_refused(manifest, cloud)

    def test_same_dimension_different_model_id_is_refused(self, synthetic):
        """R-4: dimension alone is not identity; the model id must match too.

        A fake embedder that reports the fixture's width but a different model
        name must be refused exactly like a cloud one.
        """
        manifest = gb.load_fixture_manifest(synthetic)
        impostor = KeywordEmbedder(
            model_name="bench-embedder-impostor", dimensions=SYNTH_DIM
        )
        assert impostor.dimensions == manifest["embedder"]["dimensions"]
        self._assert_all_refused(manifest, impostor)


class TestNamespace:
    def test_other_group_decoy_is_never_returned(self, synthetic, tmp_path):
        """N1: the decoy matches ``alpha`` as strongly as the real seed.

        It lives under ``group_id="other"``. If any arm forgets
        ``group_ids=["bench"]`` it becomes a seed and its episode surfaces.
        """
        results = run_fixture_arms(
            synthetic,
            ["alpha"],
            db_path=tmp_path / "db-n1",
            embedder=KeywordEmbedder(),
            doc_ids=SYNTH_DOC_IDS,
        )

        for arm_name, rows in results.items():
            returned = {chunk_id for row in rows for chunk_id, _ in row}
            assert "decoy-doc" not in returned, (
                f"{arm_name} returned the group_id='other' decoy episode"
            )

        entity_ids = {
            chunk_id
            for row in results["graph-entity"]
            for chunk_id, _ in row
        }
        assert "bench-doc-a" in entity_ids, (
            "the bench episode that mentions the matching entity was not returned; "
            "the arm is not searching the bench graph"
        )


class TestEpisodeToChunkMapping:
    def test_unknown_episode_name_is_an_error_not_a_silent_drop(self, tmp_path):
        """An episode's ``name`` is its chunk id (pre-registration, mapping).

        An episode that does not resolve to a corpus id must raise, naming the
        offending name. Dropping it quietly would let the graph arm score
        against a chunk space it cannot address.
        """
        extra_nodes = [
            {
                "kind": "node",
                "table": "Episodic",
                "row": {
                    "uuid": "ep-unknown",
                    "name": "not-a-corpus-doc",
                    "group_id": "bench",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "source": "text",
                    "source_description": "unknown",
                    "content": "unknown",
                    "valid_at": "2026-01-01T00:00:00+00:00",
                    "entity_edges": [],
                },
            }
        ]
        extra_rels = [
            {
                "kind": "rel",
                "table": "MENTIONS",
                "src_label": "Episodic",
                "src": "ep-unknown",
                "dst_label": "Entity",
                "dst": "ent-alpha",
                "row": {
                    "uuid": "m-unknown",
                    "group_id": "bench",
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
            }
        ]
        fixture = build_synthetic_fixture(
            tmp_path,
            records=synthetic_records(extra_nodes=extra_nodes, extra_rels=extra_rels),
        )

        with pytest.raises(ValueError, match="not-a-corpus-doc"):
            run_fixture_arms(
                fixture,
                ["alpha"],
                db_path=tmp_path / "db-mapping",
                embedder=KeywordEmbedder(),
                doc_ids=SYNTH_DOC_IDS,
                names=["graph-entity"],
            )
