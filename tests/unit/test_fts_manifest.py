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
    LINUX_AMD64_SHA256,
    LINUX_AMD64_SIZE,
    OSX_AMD64_SHA256,
    OSX_AMD64_SIZE,
    OSX_ARM64_SHA256,
    OSX_ARM64_SIZE,
    artifact_url,
    fts_no_pin,
    no_pin_reason,
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


def _platform_label() -> str:
    """The platform key, or the same description the message falls back to."""
    import platform

    from docugraph.storage import fts

    return fts.platform_key() or f"{platform.system()}_{platform.machine()}".lower()


def _no_pin() -> str:
    """S1's no-pin variant, built from this machine's installed version and key."""
    from docugraph.storage import fts

    return fts_no_pin(fts.package_version(), _platform_label())


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
    expected = FTS_NO_FILE if pin is not None else _no_pin()
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


def test_f4_an_approved_digest_is_used_only_while_the_bytes_are_unchanged(tmp_path, monkeypatch):
    """F4: with no pin, the approval pins the exact bytes a human accepted.

    The approval path only exists where no pin does (MED-1), so this test takes
    the pin away for the running platform instead of relying on one.
    """
    fts = _fts()
    monkeypatch.setattr(_pins(), "PINS", {})

    path = fts.cache_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unverified-but-approved")

    digest = fts.record_approval(tmp_path, path)
    assert digest == hashlib.sha256(b"unverified-but-approved").hexdigest()
    assert fts.resolve_verified_artifact(tmp_path) == path

    path.write_bytes(b"changed after approval")
    with pytest.raises(fts.FtsUnavailableError) as excinfo:
        fts.resolve_verified_artifact(tmp_path)
    assert str(excinfo.value) == _no_pin()


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


# ------------------------------------------------ MED-2 (approved sentences)


def test_med2_the_unavailable_sentence_matches_the_approved_text():
    """The three reason variants, character for character — no paraphrase."""
    fts = _fts()

    assert fts.FTS_NO_FILE == FTS_NO_FILE
    assert fts.no_pin_sentence() == _no_pin()
    assert fts.FTS_CACHED_MISMATCH == FTS_CACHED_MISMATCH


def test_med2_the_no_pin_reason_names_the_installed_version_and_platform():
    """§10.8 S1: the no-pin reason says which machine is not covered."""
    fts = _fts()
    sentence = fts.no_pin_sentence()
    reason = no_pin_reason(fts.package_version(), _platform_label())

    assert f"({reason})" in sentence


def test_med2_the_no_pin_variant_does_not_offer_fetch_fts():
    """A platform with no pin cannot verify a download, so fetch-fts is absent."""
    fts = _fts()
    sentence = fts.no_pin_sentence()

    assert "docugraph graph fetch-fts" not in sentence
    assert "docugraph graph install-fts --unverified" in sentence
    assert "graph_db back to kuzu" in sentence


# ------------------------------------- F6 / MED-1 (a pin outranks an approval)


def test_f6_a_pinned_platform_refuses_bytes_an_approval_record_names(tmp_path):
    """MED-1: an approval may not override a pin.

    The approval record names the altered bytes exactly, so the old ordering
    would load them. The pin exists for this key and does not match, so the
    file is refused.
    """
    fts = _fts()
    if (fts.compat_version(), fts.platform_key()) not in _pins().PINS:
        pytest.skip("this platform has no pin by design")

    path = fts.cache_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"altered after the pin was measured")
    fts.record_approval(tmp_path, path)

    with pytest.raises(fts.FtsUnavailableError) as excinfo:
        fts.resolve_verified_artifact(tmp_path)
    assert str(excinfo.value) == FTS_CACHED_MISMATCH


class _FakeLadybugModule:
    """Stands in for the package so ``install_unverified`` never runs INSTALL."""

    class Database:  # noqa: D106 - a stand-in, nothing to document
        def __init__(self, *args: object) -> None:
            pass

        def close(self) -> None:
            pass

    class Connection:  # noqa: D106 - a stand-in, nothing to document
        def __init__(self, *args: object) -> None:
            pass

        def execute(self, *args: object) -> None:
            pass

        def close(self) -> None:
            pass


def _fake_install(monkeypatch: pytest.MonkeyPatch, installed: object):
    import sys
    import types

    fts = _fts()
    fake = types.SimpleNamespace(
        Database=_FakeLadybugModule.Database, Connection=_FakeLadybugModule.Connection
    )
    monkeypatch.setitem(sys.modules, "ladybug", fake)
    monkeypatch.setattr(fts, "_installed_extension_path", lambda: installed)
    return fts


def test_f6b_an_unverified_install_matching_the_pin_is_verified_without_a_record(
    tmp_path, fts_artifact, monkeypatch
):
    """MED-1: on a pinned platform a matching install is verified, not recorded."""
    fts = _fts()
    if (fts.compat_version(), fts.platform_key()) not in _pins().PINS:
        pytest.skip("this platform has no pin by design")

    installed = tmp_path / "installed" / "libfts.lbug_extension"
    installed.parent.mkdir()
    installed.write_bytes(fts_artifact.read_bytes())
    _fake_install(monkeypatch, installed)

    target = fts.install_unverified(tmp_path / "data")

    assert target == fts.cache_path(tmp_path / "data")
    assert fts.sha256_file(target) == fts_artifact_sha(fts_artifact.read_bytes())
    assert not fts.approval_record_path(tmp_path / "data").exists()
    assert fts.resolve_verified_artifact(tmp_path / "data") == target


def test_f6c_an_unverified_install_mismatching_the_pin_is_deleted_and_refused(
    tmp_path, monkeypatch
):
    """MED-1: a pin mismatch deletes the copy, records nothing, names both digests."""
    fts = _fts()
    pin = _pins().PINS.get((fts.compat_version(), fts.platform_key()))
    if pin is None:
        pytest.skip("this platform has no pin by design")

    installed = tmp_path / "installed" / "libfts.lbug_extension"
    installed.parent.mkdir()
    installed.write_bytes(b"altered install")

    _fake_install(monkeypatch, installed)

    with pytest.raises(fts.FtsDigestMismatch) as excinfo:
        fts.install_unverified(tmp_path / "data")

    message = str(excinfo.value)
    assert pin["sha256"] in message
    assert hashlib.sha256(b"altered install").hexdigest() in message
    assert not installed.exists()
    assert not fts.cache_path(tmp_path / "data").exists()
    assert not fts.approval_record_path(tmp_path / "data").exists()


# ------------------------------------------------- LOW-2 (atomic replacement)


def test_low2_a_failed_replace_leaves_no_file_and_no_debris(tmp_path, fts_artifact, monkeypatch):
    """LOW-2: the target appears only when the whole file has been written."""
    fts = _fts()
    if (fts.compat_version(), fts.platform_key()) not in _pins().PINS:
        pytest.skip("this platform has no pin by design")

    def power_loss(src: object, dst: object) -> None:
        raise OSError("simulated power loss during replace")

    monkeypatch.setattr(fts.os, "replace", power_loss)

    with pytest.raises(OSError):
        fts.fetch_fts(tmp_path, downloader=lambda url: fts_artifact.read_bytes())

    target = fts.cache_path(tmp_path)
    assert not target.exists()
    leftovers = sorted(p.name for p in target.parent.iterdir())
    assert leftovers == []


def test_low1_fetch_works_when_the_data_directory_contains_a_quote(tmp_path, fts_artifact):
    """LOW-1: the load probe runs from the cache directory, quote and all."""
    fts = _fts()
    if (fts.compat_version(), fts.platform_key()) not in _pins().PINS:
        pytest.skip("this platform has no pin by design")

    data = tmp_path / "o'brien"
    payload = fts_artifact.read_bytes()

    path = fts.fetch_fts(data, downloader=lambda url: payload)

    assert path == fts.cache_path(data)
    assert "'" in str(path)
    assert fts.resolve_verified_artifact(data) == path
