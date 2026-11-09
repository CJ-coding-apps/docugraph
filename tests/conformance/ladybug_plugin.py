"""Run graphiti's own Kùzu-parameter tests against docugraph's Ladybug driver.

Loaded with ``-p ladybug_plugin`` (PYTHONPATH includes this directory) while
pytest collects graphiti v0.30.2's suite. The stand-in is installed at import
time -- before any conftest can import ``graphiti_core.driver.kuzu_driver`` --
so the real module (which imports ``kuzu``) is never loaded. No graphiti file
is edited.

ORACLE: WP-docugraph-0.4.0 §3.5 "Graphiti conformance (Ladybug)", §5 C1.
"""

from __future__ import annotations

import sys
import types

from docugraph.storage.ladybug_driver import (
    SCHEMA_QUERIES,
    LadybugDriver,
    LadybugDriverSession,
)

_stand_in = types.ModuleType("graphiti_core.driver.kuzu_driver")
_stand_in.KuzuDriver = LadybugDriver
_stand_in.KuzuDriverSession = LadybugDriverSession
_stand_in.SCHEMA_QUERIES = SCHEMA_QUERIES

sys.modules["graphiti_core.driver.kuzu_driver"] = _stand_in
