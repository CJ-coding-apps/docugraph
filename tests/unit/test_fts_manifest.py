"""The FTS artifact: pins, verification, approval records, refusals.

ORACLE: WP-docugraph-0.4.0 §10.2 D-6 (verified file by path, pins are
tamper-evident not signatures), §10.3 (the three options and the unverified
path), §10.6 F1/F3/F4.
"""

from __future__ import annotations

import hashlib

import pytest
from click.testing import CliRunner

from backend_oracle import (
    FTS_CACHED_MISMATCH,
    FTS_INSTALL_NEEDS_FLAG,
    FTS_INSTALL_NEEDS_YES,
    FTS_NO_FILE,
    FTS_NO_PIN,
    LINUX_AMD64_SHA256,
    LINUX_AMD64_SIZE,
    OSX_AMD64_SHA256,
    OSX_AMD64_SIZE,
    OSX_ARM64_SHA256,
    OSX_ARM64_SIZE,
    artifact_url,
)

# `fetch_fts` confirms a downloaded artifact by loading it in a scratch Ladybug
# database, which registers the engine's types for the rest of the process and
# makes Kùzu's extension unimportable. tests/conftest.py runs this module last.
pytestmark = pytest.mark.ladybug_engine


def _fts():
    from docugraph.storage import fts

    return fts


def _pins():
    from docugraph.storage import fts_pins

    return fts_pins


def test_platform_keys_are_the_names_ladybug_uses():
    fts = _fts()
    assert fts.platform_key("Darwin", "x86_64") == "osx_amd64"
    assert fts.platform_key("Darwin", "arm64") == "osx_arm64"
    assert fts.platform_key("Linux", "x86_64") == "linux_amd64"
    assert fts.platform_key("Linux", "aarch64") == "linux_arm64"
    # Not claimed by 0.4.0, and not silently guessed into a Linux name.
    assert fts.platform_key("Windows", "AMD64") is None


def test_the_measured_pins_are_recorded_for_the_claimed_platforms():
    """P0-4: the digests measured on 2026-11-08, keyed by extension version."""
    pins = _pins().PINS
    osx = pins[("0.22.0", "osx_amd64")]
    linux = pins[("0.22.0", "linux_amd64")]

    assert osx["sha256"] == OSX_AMD64_SHA256
    assert osx["size"] == OSX_AMD64_SIZE
    assert osx["url"] == artifact_url("osx_amd64")
    assert "0.22.3" in osx["tested_packages"]

    assert linux["sha256"] == LINUX_AMD64_SHA256
    assert linux["size"] == LINUX_AMD64_SIZE
    assert linux["url"] == artifact_url("linux_amd64")
    assert "0.22.3" in linux["tested_packages"]

    # The arm64 digests were measured but only shipped once the CI leg that
    # exercises them is green (§10.9 note 3). When it is, both the pin and this
    # assertion are added together; until then the entry must not exist.
    if ("0.22.0", "osx_arm64") in pins:
        assert pins[("0.22.0", "osx_arm64")]["sha256"] == OSX_ARM64_SHA256
        assert pins[("0.22.0", "osx_arm64")]["size"] == OSX_ARM64_SIZE
    assert ("0.22.0", "linux_arm64") not in pins


def test_the_package_version_maps_to_the_extension_compat_directory():
    assert _pins().compat_version("0.22.3") == "0.22.0"
    assert _pins().compat_version("0.22.9") == "0.22.0"
    assert _pins().compat_version("0.23.0") is None


def test_no_verified_file_names_the_reason_and_three_ways_forward(tmp_path):
    fts = _fts()
    with pytest.raises(fts.FtsUnavailableError) as excinfo:
        fts.resolve_verified_artifact(tmp_path)

    pin = _pins().PINS.get((fts.compat_version(), fts.platform_key()))
    expected = FTS_NO_FILE if pin is not None else FTS_NO_PIN
    assert str(excinfo.value) == expected


def test_f1_a_download_that_does_not_match_the_pin_is_deleted(tmp_path, monkeypatch):
    """F1: altered bytes are removed and never loaded."""
    fts = _fts()
    if (fts.compat_version(), fts.platform_key()) not in _pins().PINS:
        pytest.skip("this platform has no pin by design")
    monkeypatch.setattr(fts, "_download_bytes", lambda url: b"altered payload")

    with pytest.raises(fts.FtsDigestMismatch):
        fts.fetch_fts(tmp_path)

    assert not fts.cache_path(tmp_path).exists()


def test_fetch_records_the_pinned_file_and_resolves_it(tmp_path, fts_artifact):
    fts = _fts()
    if (fts.compat_version(), fts.platform_key()) not in _pins().PINS:
        pytest.skip("this platform has no pin by design")
    payload = fts_artifact.read_bytes()

    path = fts.fetch_fts(tmp_path, downloader=lambda url: payload)

    assert path == fts.cache_path(tmp_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == fts_artifact_sha(payload)
    assert fts.resolve_verified_artifact(tmp_path) == path


def fts_artifact_sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_f4_an_approved_digest_is_used_only_while_the_bytes_are_unchanged(tmp_path):
    """F4: an unverified install is pinned to the exact bytes a human approved."""
    fts = _fts()
    path = fts.cache_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unverified-but-approved")

    digest = fts.record_approval(tmp_path, path)
    assert digest == hashlib.sha256(b"unverified-but-approved").hexdigest()
    assert fts.resolve_verified_artifact(tmp_path) == path

    path.write_bytes(b"changed after approval")
    with pytest.raises(fts.FtsUnavailableError) as excinfo:
        fts.resolve_verified_artifact(tmp_path)
    assert str(excinfo.value) == FTS_CACHED_MISMATCH


def test_f3_install_without_yes_refuses_a_non_terminal():
    """F3: no TTY and no --yes means refuse, before any download."""
    from docugraph.interfaces.cli import main

    result = CliRunner().invoke(main, ["graph", "install-fts", "--unverified"])

    assert result.exit_code == 1
    assert FTS_INSTALL_NEEDS_YES in result.output


def test_install_fts_requires_the_unverified_flag():
    from docugraph.interfaces.cli import main

    result = CliRunner().invoke(main, ["graph", "install-fts"])

    assert result.exit_code == 1
    assert FTS_INSTALL_NEEDS_FLAG in result.output


def test_the_unverified_prompt_states_the_installer_risk():
    """§10.9 note 2: saying "unverified" is not enough; say what INSTALL runs."""
    from docugraph.storage.fts import FTS_INSTALL_WARNING

    assert "download and execute an installer in-process" in FTS_INSTALL_WARNING
    assert "INSTALL" in FTS_INSTALL_WARNING
