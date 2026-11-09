"""Compare a graphiti conformance run's JUnit XML with the Kùzu-pass baseline.

Usage: python tests/conformance/check_results.py <junit.xml>

Exits 1 and lists every baseline test that did not pass. Tests outside the
baseline are ignored: graphiti's suite has 21 pre-existing failures on Kùzu
too, and this job exists to prove the 25 do not regress. The baseline file was
recorded by running graphiti v0.30.2 on kuzu 0.11.3 on 2026-11-08.

ORACLE: WP-docugraph-0.4.0 §3.5 step 5, §5 C1, §10.5 ruling 6.7.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _baseline() -> set[str]:
    return {
        line.strip()
        for line in (HERE / "kuzu_pass_ids.txt").read_text().splitlines()
        if line.strip()
    }


def _passed(xml_path: Path) -> set[str]:
    passed: set[str] = set()
    for case in ET.parse(xml_path).iter("testcase"):
        if any(child.tag in ("failure", "error", "skipped") for child in case):
            continue
        classname = (case.get("classname") or "").replace(".", "/") + ".py"
        passed.add(f"{classname}::{case.get('name')}")
    return passed


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    expected = _baseline()
    passed = _passed(Path(argv[1]))
    missing = sorted(expected - passed)

    if missing:
        print(f"{len(missing)} of {len(expected)} Kùzu-pass baseline tests did not pass:")
        for node_id in missing:
            print(f"  {node_id}")
        return 1

    print(f"All {len(expected)} Kùzu-pass baseline tests passed under LadybugDriver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
