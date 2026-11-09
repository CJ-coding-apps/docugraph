"""Obtaining and verifying Ladybug's full-text-search extension.

Ladybug ships the FTS extension separately. Docugraph refuses to run the
Ladybug backend without it, because graphiti uses full-text search for episode
and entity retrieval, and a silently missing index would degrade search rather
than fail it. There are exactly two ways to obtain a file:

* ``fetch_fts`` downloads from the URL pinned for this Ladybug version and
  platform, checks the sha256 against ``fts_pins.PINS``, confirms that the file
  loads, and only then stores it in the cache.
* ``install_unverified`` runs Ladybug's own ``INSTALL FTS`` and records the
  digest of whatever it produced as a human-approved artifact. That digest is
  not compared to a pin, so it only proves the file has not changed since the
  approval.

A pin is tamper-evident against a recorded measurement, not a signature. It
does not authenticate the server that served the file.

The extension is database-scoped: the driver loads it once, on a setup
connection, before it builds the async connection pool.

ORACLE: WP-docugraph-0.4.0 §10.2 D-6, §10.3, §10.4 P0-4, §10.6 F1/F3/F4.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from docugraph.storage import fts_pins

FTS_FILENAME = "libfts.lbug_extension"

_DOWNLOAD_TIMEOUT_SECONDS = 60.0


def load_extension_statement(path: str | Path) -> str:
    """The one ``LOAD EXTENSION`` statement, built in one place.

    The path is a string literal in Cypher, and a data directory may legitimately
    contain a quote or a backslash (``/Users/o'brien/...``). Measured on
    Ladybug 0.22.3: an unescaped quote is a parser exception, while ``\\'`` and
    ``\\\\`` load the same file. The backslash is escaped first, or the escape
    it introduces would be escaped again.
    """
    literal = str(path).replace("\\", "\\\\").replace("'", "\\'")
    return f"LOAD EXTENSION '{literal}'"


class FtsError(Exception):
    """The FTS extension could not be obtained, verified, or loaded."""


class FtsUnavailableError(FtsError):
    """No usable extension is present; the message says how to get one."""


class FtsDigestMismatch(FtsError):
    """A downloaded artifact did not match its pin; nothing was written."""


class FtsInitializationError(FtsError):
    """An extension that resolved could not be loaded or used."""


_OPTIONS_PINNED = (
    "Run 'docugraph graph fetch-fts' to download and verify it; "
    "or run 'docugraph graph install-fts --unverified' to install it without verification; "
    "or set graph_db back to 'kuzu'."
)
# With no pin there is nothing for `fetch-fts` to verify a download against,
# so that option is not offered: it would be advice that cannot work.
_OPTIONS_UNPINNED = (
    "Run 'docugraph graph install-fts --unverified' to install it without verification; "
    "or set graph_db back to 'kuzu'."
)


def _unavailable(reason: str, *, pinned: bool = True) -> str:
    options = _OPTIONS_PINNED if pinned else _OPTIONS_UNPINNED
    return f"Full-text search is unavailable: {reason}. {options}"


FTS_NO_FILE = _unavailable(
    "no verified extension file is present for this Ladybug version and platform"
)
FTS_NO_PIN = _unavailable(
    "no checksum is pinned for this Ladybug version and platform", pinned=False
)
FTS_CACHED_MISMATCH = _unavailable("the cached extension does not match its pinned checksum")

# §10.9 note 2: saying "unverified" is not enough. In some Ladybug versions the
# INSTALL path can run an installer it downloads, so the prompt says so.
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

FTS_INIT_FAILED = (
    "Full-text search could not be initialized on the Ladybug backend: {error}. "
    "The graph was not opened. "
    "Run 'docugraph graph fetch-fts' to provide a verified extension; "
    "or set graph_db back to 'kuzu'."
)


def platform_key(system: str | None = None, machine: str | None = None) -> str | None:
    """Map ``platform.system``/``platform.machine`` to Ladybug's artifact name.

    ``None`` means "not a platform this release claims", never a guess: a
    wrong name would produce a 404 at download time or, worse, a file for
    another architecture.
    """
    system = (system if system is not None else platform.system()).lower()
    machine = (machine if machine is not None else platform.machine()).lower()
    if system == "darwin":
        return {"x86_64": "osx_amd64", "arm64": "osx_arm64"}.get(machine)
    if system == "linux":
        return {"x86_64": "linux_amd64", "amd64": "linux_amd64", "aarch64": "linux_arm64"}.get(
            machine
        )
    return None


def package_version() -> str:
    """The installed ``ladybug`` distribution version."""
    try:
        return version("ladybug")
    except PackageNotFoundError as exc:  # pragma: no cover - the caller gates this
        raise FtsError("the 'ladybug' package is not installed") from exc


def compat_version(package: str | None = None) -> str | None:
    """The extension compatibility version for this (or a given) package version."""
    return fts_pins.compat_version(package if package is not None else package_version())


def cache_path(data_dir: Path) -> Path:
    """Where a verified artifact for this version and platform lives."""
    compat = compat_version()
    plat = platform_key()
    if compat is None or plat is None:
        raise FtsUnavailableError(FTS_NO_PIN)
    return Path(data_dir) / "extensions" / "fts" / compat / plat / FTS_FILENAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def approval_record_path(data_dir: Path) -> Path:
    return Path(data_dir) / "graph" / "fts-approval.json"


def record_approval(data_dir: Path, path: Path) -> str:
    """Record the exact digest of an artifact a human chose to accept."""
    digest = sha256_file(path)
    record = {
        "sha256": digest,
        "size": path.stat().st_size,
        "platform": platform_key(),
        "compat": compat_version(),
        "source": "install-fts --unverified",
    }
    record_path = approval_record_path(data_dir)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return digest


def _read_approval(data_dir: Path) -> dict[str, Any] | None:
    record_path = approval_record_path(data_dir)
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def resolve_verified_artifact(data_dir: Path) -> Path:
    """Return the extension to load, or raise the sentence that names the fix.

    The pin decides. Where a pin exists for this version and platform, the file
    must match it; an approval record cannot override it. Where no pin exists,
    a human-approved record naming the file's exact digest is the only way in.
    Bytes nobody recorded are never accepted.
    """
    path = cache_path(data_dir)
    compat = compat_version()
    plat = platform_key()
    pin = fts_pins.PINS.get((compat, plat)) if compat is not None and plat is not None else None

    if not path.is_file():
        raise FtsUnavailableError(FTS_NO_PIN if pin is None else FTS_NO_FILE)

    digest = sha256_file(path)
    if pin is not None:
        if digest == pin["sha256"]:
            return path
        raise FtsUnavailableError(FTS_CACHED_MISMATCH)

    approval = _read_approval(data_dir)
    if approval is not None and approval.get("sha256") == digest:
        return path
    raise FtsUnavailableError(FTS_NO_PIN)


def _download_bytes(url: str) -> bytes:
    import httpx

    response = httpx.get(url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def _confirm_loads(path: Path) -> None:
    """Load the extension in a scratch database before trusting it."""
    import ladybug

    db = ladybug.Database(":memory:")
    try:
        conn = ladybug.Connection(db)
        try:
            conn.execute(load_extension_statement(path))
        finally:
            conn.close()
    finally:
        db.close()


def _write_atomic(target: Path, payload: bytes) -> None:
    """Write ``payload`` so that ``target`` is either absent or complete.

    An interrupted download must not leave a file that a later ``LOAD`` would
    trust. The temporary file lives in the same directory (so ``os.replace`` is
    atomic on one filesystem) and is removed if the replacement fails.
    """
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def fetch_fts(data_dir: Path, downloader: Callable[[str], bytes] | None = None) -> Path:
    """Download, verify, confirm and cache the pinned artifact."""
    compat = compat_version()
    plat = platform_key()
    if compat is None or plat is None:
        raise FtsUnavailableError(FTS_NO_PIN)
    pin = fts_pins.PINS.get((compat, plat))
    if pin is None:
        raise FtsUnavailableError(FTS_NO_PIN)

    payload = (downloader or _download_bytes)(pin["url"])
    digest = hashlib.sha256(payload).hexdigest()
    if digest != pin["sha256"]:
        raise FtsDigestMismatch(
            f"The FTS extension downloaded from {pin['url']} does not match its pinned "
            f"checksum (expected {pin['sha256']}, got {digest}). Nothing was written."
        )

    target = cache_path(data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Confirm the bytes load before caching them. The probe keeps the expected
    # filename: the loader decides by extension name, not content.
    try:
        with tempfile.TemporaryDirectory(dir=target.parent) as probe_dir:
            probe = Path(probe_dir) / FTS_FILENAME
            probe.write_bytes(payload)
            _confirm_loads(probe)
    except Exception as exc:
        raise FtsInitializationError(
            f"The FTS extension downloaded from {pin['url']} matches its pinned checksum "
            f"but did not load: {exc}"
        ) from exc
    _write_atomic(target, payload)
    return target


def _installed_extension_path() -> Path:
    """Find the file Ladybug's own INSTALL left in its extension directory."""
    compat = compat_version()
    plat = platform_key()
    if compat is None or plat is None:
        raise FtsUnavailableError(FTS_NO_PIN)
    root = Path.home() / ".lbdb" / "extension" / compat / plat
    candidates = sorted(root.rglob(FTS_FILENAME), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FtsError(
            "Ladybug's INSTALL did not leave an FTS extension in "
            f"{root}. Check the INSTALL output, then run 'docugraph graph fetch-fts'."
        )
    return candidates[-1]


def install_unverified(data_dir: Path) -> Path:
    """Run Ladybug's ``INSTALL FTS``, then apply the pin or record an approval.

    On a platform with a pin, the pin still decides: a matching install counts
    as verified and no record is written, and a mismatch deletes the installed
    copy, records nothing, and raises naming both digests. The approval record
    exists only for a key with no pin.
    """
    import ladybug

    with tempfile.TemporaryDirectory(prefix="docugraph-fts-install-") as scratch:
        db = ladybug.Database(str(Path(scratch) / "scratch.lbdb"))
        try:
            conn = ladybug.Connection(db)
            try:
                conn.execute("INSTALL FTS;")
            finally:
                conn.close()
        finally:
            db.close()

    installed = _installed_extension_path()
    compat = compat_version()
    plat = platform_key()
    pin = fts_pins.PINS.get((compat, plat)) if compat is not None and plat is not None else None

    if pin is not None:
        digest = sha256_file(installed)
        if digest != pin["sha256"]:
            installed.unlink(missing_ok=True)
            raise FtsDigestMismatch(
                f"The FTS extension Ladybug installed does not match the pinned checksum "
                f"for {plat} (expected {pin['sha256']}, got {digest}). The installed copy "
                "was deleted and nothing was recorded."
            )

    payload = installed.read_bytes()
    target = cache_path(data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(target, payload)
    if pin is None:
        record_approval(data_dir, target)
    return target
