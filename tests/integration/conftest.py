"""Shared fixtures for the browser-driven integration modules.

Chromium is probed in a subprocess, not in this interpreter: an in-process
``sync_playwright()`` probe leaves "Task was destroyed but it is pending!" at
interpreter exit. The probe is lazy -- nothing starts during collection -- and
runs only when a test that uses the fixture is set up. ``DOCUGRAPH_REQUIRE_BROWSER=1``
turns a missing browser into a failure instead of a skip (the CI job sets it).
"""

import os
import subprocess
import sys

import pytest

_PROBE = """
from playwright.sync_api import sync_playwright

with sync_playwright() as playwright:
    print(playwright.chromium.executable_path)
"""


def chromium_installed(*, timeout: float = 60.0) -> bool:
    """True when Playwright's Chromium executable is present on this machine."""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:  # pragma: no cover - a broken install is a skip/failure
        return False
    if completed.returncode != 0:
        return False
    path = completed.stdout.strip()
    return bool(path) and os.path.exists(path)


@pytest.fixture(scope="session")
def require_chromium() -> None:
    """Skip when Chromium is missing; fail instead if CI requires it."""
    if chromium_installed():
        return
    message = "Chromium is not installed; run `uv run playwright install chromium`"
    if os.environ.get("DOCUGRAPH_REQUIRE_BROWSER") == "1":
        pytest.fail(message, pytrace=False)
    pytest.skip(message)
