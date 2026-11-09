"""The conformance run cannot pass through the wrong driver.

Run alongside graphiti's suite by the `Graphiti conformance (Ladybug)` job,
with the plugin active. Skipped everywhere else.

ORACLE: WP-docugraph-0.4.0 §3.5 step 3, §5 C1.
"""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DOCUGRAPH_LADYBUG_CONFORMANCE") != "1",
    reason="only meaningful in the conformance job with ladybug_plugin loaded",
)


def test_the_module_graphiti_imports_is_docugraphs_driver():
    from docugraph.storage.ladybug_driver import LadybugDriver

    module = sys.modules["graphiti_core.driver.kuzu_driver"]
    assert module.KuzuDriver is LadybugDriver

    driver = module.KuzuDriver(db=":memory:")
    assert type(driver).__module__ == "docugraph.storage.ladybug_driver"
    assert "kuzu" not in sys.modules
