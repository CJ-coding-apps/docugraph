"""Pinned full-text-search extension artifacts.

An entry is keyed by ``(extension compatibility version, platform)``. The
compatibility version is Ladybug's own, not the package version: 0.22.3 serves
its extension from a ``v0.22.0`` directory, and every 0.22.x package is expected
to share it. The platform names are the ones in the artifact URL.

What a pin is: a checksum measured from one download on 2026-11-08 (WP-0.4.0
Phase 0). It makes a download tamper-evident against that measurement; it is
not a signature and it does not authenticate the server. CI re-downloads the
artifact on every run and fails if the digest moves, which is the only
independent check behind these numbers.

Adding a platform requires an entry here; ``fts.cache_path`` and
``fts.fetch_fts`` refuse an unpinned combination rather than guess a URL.

ORACLE: WP-docugraph-0.4.0 §10.2 D-6, §10.4 P0-4, §10.9 note 3.
"""

from __future__ import annotations

from typing import Any

# Ladybug's extension compatibility version for each package minor.
_COMPAT_BY_MINOR: dict[str, str] = {
    "0.22": "0.22.0",
}

PINS: dict[tuple[str, str], dict[str, Any]] = {
    ("0.22.0", "osx_amd64"): {
        "url": "https://extension.ladybugdb.com/v0.22.0/osx_amd64/fts/libfts.lbug_extension",
        "sha256": "a016f7d56611a7218d6ac2f4cace083ec414e39ea9ceb601e44233c1f9fa7b4a",
        "size": 1_640_512,
        "tested_packages": ("0.22.3",),
    },
    # Measured from the same artifact URL, not from a run on this machine: the
    # digest is the file's, independent of the host that downloads it. The
    # `macos-15` (arm64) CI leg is what proves the file loads and runs there;
    # if that leg does not pass, both the leg and this entry are removed
    # before this change is handed back. `linux_arm64` stays unpinned: no CI
    # runner here can exercise it.
    ("0.22.0", "osx_arm64"): {
        "url": "https://extension.ladybugdb.com/v0.22.0/osx_arm64/fts/libfts.lbug_extension",
        "sha256": "6fdf9749f47ca2e35c82302055ff441061a68158dfb2c4a61c396ccd6dabe289",
        "size": 1_628_824,
        "tested_packages": ("0.22.3",),
    },
    ("0.22.0", "linux_amd64"): {
        "url": "https://extension.ladybugdb.com/v0.22.0/linux_amd64/fts/libfts.lbug_extension",
        "sha256": "8fcc16c1f26ddea668ca76bc18f6f83a1588238a4baa08511106baa51e5f4430",
        "size": 2_305_304,
        "tested_packages": ("0.22.3",),
    },
}


def compat_version(package_version: str) -> str | None:
    """Map a ``ladybug`` package version to its extension compatibility version."""
    minor = ".".join(package_version.split(".")[:2])
    return _COMPAT_BY_MINOR.get(minor)
