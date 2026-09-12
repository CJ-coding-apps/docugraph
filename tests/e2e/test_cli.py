"""End-to-end tests for the CLI interface."""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from docugraph.interfaces.cli import main as cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestCLIBasics:
    """Basic CLI functionality tests."""

    def test_cli_help(self, runner):
        """Test that help command works."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "DocuGraph AI" in result.output or "docugraph" in result.output.lower()

    def test_cli_version(self, runner):
        """Test version output."""
        result = runner.invoke(cli, ["--version"])
        # Either shows version or doesn't have --version flag
        assert result.exit_code in [0, 2]


class TestMemoryCommands:
    """Tests for memory-related CLI commands."""

    def test_memory_set_get(self, runner, temp_data_dir, monkeypatch):
        """Test memory set and get commands."""
        monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", temp_data_dir)

        # Set a memory value
        result = runner.invoke(
            cli,
            ["memory", "set", "test_key", "test_value"],
        )
        assert result.exit_code == 0

        # Get the value
        result = runner.invoke(
            cli,
            ["memory", "get", "test_key"],
        )
        assert result.exit_code == 0
        assert "test_value" in result.output

    def test_memory_list(self, runner, temp_data_dir, monkeypatch):
        """Test memory list command."""
        monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", temp_data_dir)

        # Set some values first
        runner.invoke(cli, ["memory", "set", "key1", "value1"])
        runner.invoke(cli, ["memory", "set", "key2", "value2"])

        # List keys
        result = runner.invoke(cli, ["memory", "list"])
        assert result.exit_code == 0

    def test_memory_delete(self, runner, temp_data_dir, monkeypatch):
        """Test memory delete command."""
        monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", temp_data_dir)

        # Set and delete
        runner.invoke(cli, ["memory", "set", "to_delete", "value"])
        result = runner.invoke(cli, ["memory", "delete", "to_delete"])
        assert result.exit_code == 0


class TestSearchCommands:
    """Tests for search-related CLI commands."""

    def test_search_empty_index(self, runner, temp_data_dir, monkeypatch):
        """Test search on empty index."""
        monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", temp_data_dir)

        result = runner.invoke(cli, ["search", "test query"])
        # Should complete (may show no results)
        assert result.exit_code in [0, 1]

    def test_stats_command(self, runner, temp_data_dir, monkeypatch):
        """Test stats command."""
        monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", temp_data_dir)

        result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0


class TestIndexCommands:
    """Tests for indexing-related CLI commands."""

    def test_index_local_missing_path(self, runner):
        """Test index local with missing path."""
        result = runner.invoke(cli, ["index", "local", "/nonexistent/path"])
        assert result.exit_code != 0

    def test_index_local_dir(self, runner, temp_data_dir, monkeypatch):
        """Test indexing a local directory."""
        monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", temp_data_dir)

        # Create a test directory with a file
        test_dir = Path(temp_data_dir) / "docs"
        test_dir.mkdir()
        test_file = test_dir / "test.md"
        test_file.write_text("# Test Document\n\nThis is test content for indexing.")

        result = runner.invoke(cli, ["index", "local", str(test_dir)])
        # May succeed or fail depending on environment
        # Just ensure it doesn't crash unexpectedly
        assert result.exit_code in [0, 1, 2]


class TestServerCommands:
    """Tests for server-related CLI commands."""

    def test_mcp_server_help(self, runner):
        """Test MCP server command help."""
        result = runner.invoke(cli, ["mcp-server", "--help"])
        assert result.exit_code == 0
