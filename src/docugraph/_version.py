"""The package version, read from installed distribution metadata.

`pyproject.toml` is the one home for the version number. Everything that
reports it -- ``__version__``, the CLI's ``--version``, the MCP handshake's
``serverInfo.version``, and the crawler's User-Agent -- reads it from here,
so the number cannot drift into a second place and start disagreeing with
itself. (It did: the distribution said 1.0.0, the CLI said 0.1.0, and the MCP
handshake reported the *mcp SDK's* version because the server was constructed
without one.)

``DISTRIBUTION_NAME`` must match ``name`` in pyproject.toml; a test asserts
that, so renaming the package fails loudly here instead of silently falling
back to the placeholder below.
"""

from importlib.metadata import PackageNotFoundError, version

DISTRIBUTION_NAME = "docugraph-ai-v1"

_UNKNOWN = "0.0.0.dev0"

try:
    __version__ = version(DISTRIBUTION_NAME)
except PackageNotFoundError:  # pragma: no cover -- source tree, never installed
    # Running from a checkout that was never installed (no metadata to read).
    # A placeholder is better than crashing on import, and better than a
    # plausible-looking number that would be a lie.
    __version__ = _UNKNOWN
