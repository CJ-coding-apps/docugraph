"""The exact sentences and constants the 0.4.0 backend contract promises.

Kept in one place so a test can assert a user-visible string character for
character rather than a paraphrase. The implementation imports nothing from
here; these are the oracle, not the source.

ORACLE: WP-docugraph-0.4.0 §3.2 (D-3), §10.3 (D-5), §10.2 (D-9).
"""

from __future__ import annotations

# §3.2, verbatim and complete.
GRAPH_NEEDS_MIGRATION_MESSAGE = (
    "This graph was created with the Kùzu backend. To use Ladybug, migrate it first: "
    "docugraph graph migrate-to-ladybug (or set graph_db back to kuzu)."
)


# §10.8 S1, verbatim.
_S1_OPTIONS_PINNED = (
    "Run: docugraph graph fetch-fts — or approve an unverified install: "
    "docugraph graph install-fts --unverified — or set graph_db back to kuzu."
)
# No pin means `fetch-fts` cannot verify anything, so the approved wording drops
# that clause and rephrases the remaining two options.
_S1_OPTIONS_UNPINNED = (
    "Run: docugraph graph install-fts --unverified to approve an unverified install "
    "— or set graph_db back to kuzu."
)


def fts_unavailable(reason: str, *, pinned: bool = True) -> str:
    """The one sentence that names the reason and the ways forward.

    ``pinned=False`` is the no-checksum variant: the fetch clause is omitted
    because ``fetch-fts`` has nothing to compare a download against.
    """
    options = _S1_OPTIONS_PINNED if pinned else _S1_OPTIONS_UNPINNED
    return (
        "The Ladybug backend needs its full-text search extension, "
        f"and no verified copy is available ({reason}). {options}"
    )


NO_FILE_REASON = "not downloaded yet"
CACHED_MISMATCH_REASON = "the cached file does not match its pinned checksum"


def no_pin_reason(version: str, platform: str) -> str:
    """The no-pin reason names the installed package version and platform key."""
    return f"no pinned checksum for ladybug {version} on {platform}"


FTS_NO_FILE = fts_unavailable(NO_FILE_REASON)
FTS_CACHED_MISMATCH = fts_unavailable(CACHED_MISMATCH_REASON)


def fts_no_pin(version: str, platform: str) -> str:
    return fts_unavailable(no_pin_reason(version, platform), pinned=False)


# §10.3 (b): the prompt must state this, not merely say "unverified".
FTS_INSTALL_WARNING = (
    "This installs Ladybug's FTS extension without a checksum to check it against. "
    "Ladybug's INSTALL can, in some versions, download and execute an installer in-process. "
    "Only continue if you trust the extension repository and your connection to it."
)

FTS_INSTALL_NEEDS_YES = (
    "Refusing to install the unverified FTS extension without confirmation: "
    "this is not a terminal, so pass --yes to proceed."
)

FTS_INSTALL_NEEDS_FLAG = (
    "graph install-fts installs the FTS extension without verification; "
    "pass --unverified to acknowledge that, or use 'graph fetch-fts'."
)

# §10.8 S2, verbatim: a failed load or create is an initialization error that
# says the graph was not opened and names the fix, including the Kùzu fallback.
FTS_INIT_FAILED = (
    "The Ladybug backend could not set up full-text search ({error}); "
    "the graph was not opened. "
    "Run docugraph graph fetch-fts to re-verify the extension, "
    "or set graph_db back to kuzu."
)

# §10.4 P0-4, measured on 2026-11-08 (macOS 15 x86_64 and python:3.11-slim).
OSX_AMD64_SHA256 = "a016f7d56611a7218d6ac2f4cace083ec414e39ea9ceb601e44233c1f9fa7b4a"
OSX_AMD64_SIZE = 1_640_512
LINUX_AMD64_SHA256 = "8fcc16c1f26ddea668ca76bc18f6f83a1588238a4baa08511106baa51e5f4430"
LINUX_AMD64_SIZE = 2_305_304
OSX_ARM64_SHA256 = "6fdf9749f47ca2e35c82302055ff441061a68158dfb2c4a61c396ccd6dabe289"
OSX_ARM64_SIZE = 1_628_824


def artifact_url(platform: str, compat: str = "0.22.0") -> str:
    return f"https://extension.ladybugdb.com/v{compat}/{platform}/fts/libfts.lbug_extension"
