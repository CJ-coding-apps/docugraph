"""The browser requirement drives a child pytest, never this interpreter.

Each row starts a subprocess so the parent's interpreter is never the one
probing Playwright. The second row is the regression for the pending-task
warning: collecting the integration tree must print nothing at exit.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_pytest(args: list[str], env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, **env_extra},
    )


def test_i1_require_browser_turns_a_missing_chromium_into_a_failure(tmp_path) -> None:
    """ORACLE: A-2 / I1 — DOCUGRAPH_REQUIRE_BROWSER=1 fails, it does not skip."""
    empty_browsers = tmp_path / "no-browsers"
    empty_browsers.mkdir()
    completed = _run_pytest(
        [
            "-q",
            "tests/integration/test_crawl_redirect.py"
            "::test_c3_redirect_to_a_blocked_host_is_never_fetched",
        ],
        {
            "DOCUGRAPH_REQUIRE_BROWSER": "1",
            "PLAYWRIGHT_BROWSERS_PATH": str(empty_browsers),
        },
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert "Chromium" in output, output
    assert "playwright install chromium" in output, output


def test_i2_collecting_the_integration_tree_says_nothing_at_exit() -> None:
    """ORACLE: A-2 / I2 — collection never probes Playwright in-process."""
    completed = _run_pytest(["--collect-only", "-q", "tests/integration"], {})

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "Task was destroyed" not in output, output
