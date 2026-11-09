"""The Ladybug driver: Kùzu's API, Ladybug's binding rule, no Kùzu import.

ORACLE: WP-docugraph-0.4.0 §3.1, §5 L1/L2/L3 (driver half).

The driver is derived from graphiti v0.30.2's Kùzu driver (Apache-2.0). It
reports ``GraphProvider.KUZU`` so graphiti's Kùzu operations run unchanged, but
it imports ``ladybug`` and keeps every caller-supplied parameter -- including
``None`` -- because Ladybug raises when a query references a key the dict omits
where Kùzu bound NULL.
"""

from __future__ import annotations

import pytest

# Constructing a Ladybug database makes Kùzu's extension unimportable for the
# rest of the process; tests/conftest.py runs this module last for that reason.
pytestmark = pytest.mark.ladybug_engine


def test_provider_is_kuzu_so_graphiti_takes_its_kuzu_paths():
    from graphiti_core.driver.driver import GraphProvider

    from docugraph.storage.ladybug_driver import LadybugDriver, LadybugDriverSession

    assert LadybugDriver.provider is GraphProvider.KUZU
    assert LadybugDriverSession.provider is GraphProvider.KUZU
    assert LadybugDriver.__module__ == "docugraph.storage.ladybug_driver"


def test_importing_the_driver_does_not_require_kuzu(run_blocked):
    """L3 (import half): with kuzu blocked, the driver still imports."""
    proc, payload = run_blocked(
        """
from docugraph.storage.ladybug_driver import LadybugDriver

print(json.dumps({
    "provider": str(LadybugDriver.provider),
    "kuzu_driver_imported": "graphiti_core.driver.kuzu_driver" in sys.modules,
    "kuzu_imported": "kuzu" in sys.modules,
}))
""",
        roots=("kuzu",),
    )
    assert proc.returncode == 0, proc.stderr
    assert payload == {
        "provider": "GraphProvider.KUZU",
        "kuzu_driver_imported": False,
        "kuzu_imported": False,
    }


async def test_l1_none_is_bound_as_null_not_dropped(tmp_path):
    """L1: `{'value': 'v', 'expired_at': None}` runs and binds NULL."""
    from docugraph.storage.ladybug_driver import LadybugDriver

    driver = LadybugDriver(db=str(tmp_path / "l1.lbdb"))
    try:
        await driver.execute_query(
            "CREATE NODE TABLE T(id STRING, expired_at TIMESTAMP, PRIMARY KEY(id));"
        )
        await driver.execute_query(
            "CREATE (:T {id: $id, expired_at: $expired_at});",
            id="v",
            expired_at=None,
        )
        records, _, _ = await driver.execute_query(
            "MATCH (t:T) RETURN t.id AS id, t.expired_at AS expired_at;"
        )
    finally:
        await driver.close()

    assert records == [{"id": "v", "expired_at": None}]


async def test_l2_an_omitted_parameter_is_not_invented(tmp_path):
    """L2: graphiti's old None-filter turned an omission into this error.

    The driver must pass the caller's dict through, so an omission stays an
    omission and Ladybug says so. Fabricating a NULL here would hide a caller
    bug -- and re-introducing the filtering is what mutation check 1 catches.
    """
    from docugraph.storage.ladybug_driver import LadybugDriver

    driver = LadybugDriver(db=str(tmp_path / "l2.lbdb"))
    try:
        await driver.execute_query(
            "CREATE NODE TABLE T(id STRING, expired_at TIMESTAMP, PRIMARY KEY(id));"
        )
        with pytest.raises(Exception) as excinfo:
            await driver.execute_query(
                "MATCH (t:T) WHERE t.expired_at = $expired_at RETURN t.id;",
                id="v",
            )
    finally:
        await driver.close()

    assert "expired_at" in str(excinfo.value)


async def test_an_unreferenced_extra_parameter_is_ignored(tmp_path):
    """Ladybug accepts extra keys (Kùzu did not); the driver adds nothing."""
    from docugraph.storage.ladybug_driver import LadybugDriver

    driver = LadybugDriver(db=str(tmp_path / "extra.lbdb"))
    try:
        driver.setup_schema()
        records, _, _ = await driver.execute_query("RETURN 1 AS one;", unused="x")
    finally:
        await driver.close()

    assert records == [{"one": 1}]


async def test_database_parameters_are_not_forwarded(tmp_path):
    """graphiti passes `database_`/`routing_` kwargs; Kùzu rejects them."""
    from docugraph.storage.ladybug_driver import LadybugDriver

    driver = LadybugDriver(db=str(tmp_path / "routing.lbdb"))
    try:
        records, _, _ = await driver.execute_query(
            "RETURN $value AS value;", value="v", database_="", routing_="r"
        )
    finally:
        await driver.close()

    assert records == [{"value": "v"}]


async def test_session_reuses_the_driver(tmp_path):
    from docugraph.storage.ladybug_driver import LadybugDriver, LadybugDriverSession

    driver = LadybugDriver(db=str(tmp_path / "session.lbdb"))
    try:
        session = driver.session()
        assert isinstance(session, LadybugDriverSession)
        assert session.driver is driver
        await session.run("RETURN 1;")
    finally:
        await driver.close()


# ----------------------------------- LOW-1 (a quote or backslash in the path)


def test_low1_the_load_statement_escapes_a_quote_and_a_backslash():
    """A data directory may be called "O'Brien"; the statement must still parse.

    Ladybug's Cypher string literal honours backslash escapes, measured on
    0.22.3: an unescaped quote is a parser exception, ``\\'`` and ``\\\\`` load
    the same file. Refusing the path would leave such a user with no way to
    run the backend at all.
    """
    from docugraph.storage.ladybug_driver import load_extension_statement

    assert load_extension_statement("/tmp/o'brien/libfts.lbug_extension") == (
        "LOAD EXTENSION '/tmp/o\\'brien/libfts.lbug_extension'"
    )
    assert load_extension_statement("/tmp/a\\b/libfts.lbug_extension") == (
        "LOAD EXTENSION '/tmp/a\\\\b/libfts.lbug_extension'"
    )
    assert load_extension_statement("/tmp/plain/libfts.lbug_extension") == (
        "LOAD EXTENSION '/tmp/plain/libfts.lbug_extension'"
    )


async def test_low1_an_artifact_under_a_quote_directory_actually_loads(tmp_path, fts_artifact):
    """The escaped statement is not just well-shaped: the extension loads."""
    import shutil

    from docugraph.storage.fts import FTS_FILENAME
    from docugraph.storage.ladybug_driver import LadybugDriver

    target = tmp_path / "o'brien" / FTS_FILENAME
    target.parent.mkdir()
    shutil.copyfile(fts_artifact, target)

    driver = LadybugDriver(db=str(tmp_path / "quote.lbdb"), fts_extension=str(target))
    try:
        records, _, _ = await driver.execute_query("RETURN 1 AS one;")
    finally:
        await driver.close()

    assert records == [{"one": 1}]
