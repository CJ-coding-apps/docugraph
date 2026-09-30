"""The version number has exactly one home, and everything reports it.

The defect these guard against: the distribution said 1.0.0, the CLI said
0.1.0, and the MCP handshake advertised 1.30.0 -- which was the *mcp* SDK's
version, because the server was constructed without one and the SDK falls back
to `pkg_version("mcp")`. Three numbers for one package, none of them from the
same place. An MCP client that checks a server's version on connect (maf does)
had nothing real to check.
"""

import tomllib
from importlib.metadata import version
from pathlib import Path

from click.testing import CliRunner

from docugraph import __version__
from docugraph._version import DISTRIBUTION_NAME
from docugraph.core.config import CrawlerConfig
from docugraph.interfaces.cli import main as cli
from docugraph.interfaces.mcp_server import server

REPO_ROOT = Path(__file__).resolve().parents[2]


def _metadata_version() -> str:
    return version(DISTRIBUTION_NAME)


class TestSingleVersionSource:
    def test_package_version_comes_from_metadata(self):
        assert __version__ == _metadata_version()
        assert __version__ != "0.0.0.dev0", "metadata not found; is the package installed?"

    def test_distribution_name_matches_pyproject(self):
        """Renaming the package must fail here, not silently fall back."""
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        assert pyproject["project"]["name"] == DISTRIBUTION_NAME

    def test_pyproject_version_matches_metadata(self):
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        assert pyproject["project"]["version"] == _metadata_version()

    def test_cli_version_reports_the_metadata_version(self):
        result = CliRunner().invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert _metadata_version() in result.output

    def test_user_agent_carries_the_metadata_version(self):
        assert CrawlerConfig().user_agent == f"DocuGraph-AI/{_metadata_version()}"
        # A hardcoded second copy would keep reporting its own number.
        assert CrawlerConfig().user_agent.endswith(f"/{__version__}")


class TestMcpHandshakeVersion:
    def test_server_advertises_the_package_version(self):
        """The value an MCP client sees in `serverInfo.version` on connect.

        Without an explicit `version=`, this is the mcp SDK's version rather
        than ours -- the exact bug this pins.
        """
        options = server.create_initialization_options()
        assert options.server_version == _metadata_version()

    def test_server_name_is_stable(self):
        options = server.create_initialization_options()
        assert options.server_name == "docugraph"
