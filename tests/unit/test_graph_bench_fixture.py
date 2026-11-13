"""The fixture manifest (F2) and the export/import round trip (R1).

Both run on the synthetic fixture, so they need the real committed graph, an
embedding model, an LLM key, or the network. The real-fixture tests (D1, D4,
F1) live in ``test_graph_bench_real_fixture.py`` with the staging rule.
"""

import asyncio
import json

import pytest

from docugraph.evals import graph_bench as gb

from .graph_bench_helpers import (
    SYNTH_DOC_IDS,
    KeywordEmbedder,
    build_synthetic_fixture,
    canonical_rankings,
    copy_fixture,
    make_raising_llm,
    normalise_export_value,
    record_key,
    run_fixture_arms,
)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return build_synthetic_fixture(tmp_path_factory.mktemp("graph-bench-fixture"))


class TestManifestIntegrity:
    def test_manifest_graphiti_version_must_match_the_installed_one(
        self, synthetic, tmp_path
    ):
        """F2: a version mismatch fails, naming both versions."""
        tampered = copy_fixture(synthetic, tmp_path / "graph_fixture")
        manifest_path = tampered / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        installed = manifest["graphiti_version"]
        manifest["graphiti_version"] = "0.0.0-not-installed"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        problems = gb.verify_fixture_manifest(tampered)
        joined = " ".join(problems)
        assert "0.0.0-not-installed" in joined, (
            f"the manifest's version is missing from the problems: {problems}"
        )
        assert installed in joined, (
            f"the installed version {installed!r} is not named in the problems: {problems}"
        )


class TestExportImportRoundTrip:
    def test_every_field_survives_and_rankings_are_identical(self, synthetic, tmp_path):
        """R1: export -> import preserves embeddings and temporal fields.

        The input embeddings are dyadic rationals (0.5, 1.0, ...), exactly
        representable in Kùzu's FLOAT[], so an exact comparison is meaningful:
        any re-embed at import would change the exported values.
        """
        from docugraph.storage.graph_store import GraphStore

        embedder = KeywordEmbedder()
        exported = tmp_path / "exported"
        loop = asyncio.new_event_loop()
        store = GraphStore(
            tmp_path / "source_db",
            llm_client=make_raising_llm(),
            embedder=embedder,
        )
        try:
            loop.run_until_complete(gb.import_graph_fixture(store, synthetic))
            loop.run_until_complete(gb.export_graph_fixture(store, exported))
        finally:
            loop.run_until_complete(store.close())
            loop.close()

        assert (exported / "graph.jsonl").is_file(), "export wrote no graph.jsonl"
        assert (exported / "schema.cypher").is_file(), "export wrote no schema.cypher"
        # The arms only need the manifest's embedder identity; the exported
        # graph shares the source fixture's, so reuse it for the ranking check.
        (exported / "MANIFEST.json").write_text(
            (synthetic / "MANIFEST.json").read_text(encoding="utf-8"), encoding="utf-8"
        )

        source = {
            record_key(record): record for record in gb.load_graph_fixture(synthetic)
        }
        round_trip = {
            record_key(record): record for record in gb.load_graph_fixture(exported)
        }
        assert set(round_trip) == set(source), (
            "export/import changed the record set: "
            f"missing={sorted(set(source) - set(round_trip))}, "
            f"extra={sorted(set(round_trip) - set(source))}"
        )

        for key, expected_record in source.items():
            actual_record = round_trip[key]
            for field, expected_value in expected_record["row"].items():
                actual_value = actual_record["row"].get(field)
                assert normalise_export_value(actual_value) == normalise_export_value(
                    expected_value
                ), f"{key} field {field!r}: {actual_value!r} != {expected_value!r}"

        before = run_fixture_arms(
            synthetic,
            ["alpha", "gamma"],
            db_path=tmp_path / "db-before",
            embedder=embedder,
            doc_ids=SYNTH_DOC_IDS,
        )
        after = run_fixture_arms(
            exported,
            ["alpha", "gamma"],
            db_path=tmp_path / "db-after",
            embedder=embedder,
            doc_ids=SYNTH_DOC_IDS,
        )
        assert canonical_rankings(before) == canonical_rankings(after), (
            "the imported graph ranks differently from the original"
        )
