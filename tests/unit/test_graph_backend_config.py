"""The graph backend setting: two values, one default, one clear refusal.

ORACLE: WP-docugraph-0.4.0 §3.2 (the unknown value is a configuration error
naming both valid values), §5 K2, §10.5 ruling 6.5 (the CLI and MCP show one
sentence, never a traceback).
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from docugraph.core.config import Config, GraphBackend, StorageConfig, reset_config


def test_kuzu_is_the_default():
    """D-1: nothing changes for anyone who does not opt in."""
    assert StorageConfig().graph_db is GraphBackend.KUZU
    assert StorageConfig().graph_db == "kuzu"


def test_ladybug_is_accepted():
    assert StorageConfig(graph_db="ladybug").graph_db is GraphBackend.LADYBUG


def test_an_unknown_backend_names_both_valid_values():
    """K2: the error names `kuzu` and `ladybug`, so the fix is in the message."""
    with pytest.raises(ValidationError) as excinfo:
        StorageConfig(graph_db="neo4j")

    text = str(excinfo.value)
    assert "kuzu" in text
    assert "ladybug" in text


def test_the_environment_is_validated_too(monkeypatch):
    monkeypatch.setenv("DOCUGRAPH_STORAGE__GRAPH_DB", "neo4j")
    with pytest.raises(ValidationError) as excinfo:
        Config()
    text = str(excinfo.value)
    assert "kuzu" in text and "ladybug" in text


def test_the_cli_shows_a_configuration_error_without_a_traceback(monkeypatch, tmp_path):
    """Ruling 6.5: the user sees one sentence, not pydantic's frame dump."""
    from docugraph.interfaces.cli import main

    monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DOCUGRAPH_STORAGE__GRAPH_DB", "neo4j")
    reset_config()
    try:
        result = CliRunner().invoke(main, ["graph", "search", "anything"])
    finally:
        reset_config()

    output = result.output + (getattr(result, "stderr", "") or "")
    assert result.exit_code == 1
    assert "kuzu" in output and "ladybug" in output
    assert "Traceback" not in output
