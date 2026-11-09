"""The install shape and metadata, read from pyproject.toml.

ORACLE: WP-docugraph-0.3.0 §3.B1 (extras split), §3.B3 (caps), §3.B5 (PEP 639,
classifiers, URLs).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

CORE_DEPENDENCIES = {
    "pydantic",
    "pydantic-settings",
    "pyyaml",
    "rich",
    "fastembed",
    "onnxruntime",
    "lancedb",
    "pyarrow",
    "pandas",
    "click",
    "mcp",
    "httpx",
    "aiofiles",
    "tenacity",
    "xxhash",
}

EXTRA_DEPENDENCIES = {
    "crawl": {"crawl4ai", "playwright"},
    "graph": {"graphiti-core", "kuzu"},
}

REMOVED_NAMES = {"typer", "shellingham"}


def _project() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]


def _name(spec: str) -> str:
    """The distribution name of a requirement string (extras/caps stripped)."""
    return re.split(r"[\[<>=!~; ]", spec.strip(), maxsplit=1)[0].lower()


def _specs(specs: list[str]) -> dict[str, str]:
    return {_name(spec): spec for spec in specs}


def test_core_dependencies_are_the_declared_set():
    deps = _specs(_project()["dependencies"])
    assert set(deps) == CORE_DEPENDENCIES
    assert REMOVED_NAMES.isdisjoint(deps)


def test_extras_split_moves_crawl_and_graph_out_of_core():
    extras = _project()["optional-dependencies"]
    for extra, names in EXTRA_DEPENDENCIES.items():
        assert set(_specs(extras[extra])) == names


def test_all_composes_the_user_facing_extras():
    extras = _project()["optional-dependencies"]
    assert extras["all"] == ["docugraph[crawl,graph,cloud,ollama]"]


def test_dependency_caps_are_declared():
    project = _project()
    core = _specs(project["dependencies"])
    extras = project["optional-dependencies"]

    assert "<26" in core["pyarrow"]
    assert "<0.9" in core["fastembed"]
    assert "<4" in core["pandas"]
    assert "<1.24" in core["onnxruntime"]
    assert "<0.26" in core["lancedb"]
    assert "<0.10" in _specs(extras["crawl"])["crawl4ai"]
    assert "<1.63" in _specs(extras["crawl"])["playwright"]


def test_license_is_a_pep_639_expression():
    project = _project()
    assert project["license"] == "Apache-2.0"
    assert not any(c.startswith("License ::") for c in project["classifiers"])


def test_classifiers_name_the_supported_platforms_and_pythons():
    classifiers = set(_project()["classifiers"])
    required = {
        "Environment :: Console",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development",
    }
    assert required <= classifiers

    python_versions = {
        c.rsplit(" :: ", 1)[-1]
        for c in classifiers
        if c.startswith("Programming Language :: Python :: 3.")
    }
    assert python_versions == {"3.11", "3.12", "3.13"}


def test_requires_python_is_unchanged():
    assert _project()["requires-python"] == ">=3.11,<3.14"


def test_p1_ladybug_is_an_opt_in_extra_and_all_excludes_it():
    """ORACLE P1: [graph] unchanged, [ladybug] capped, [all] still Kùzu-only."""
    project = _project()
    extras = project["optional-dependencies"]

    assert extras["ladybug"] == ["ladybug>=0.22.3,<0.23"]
    assert _specs(extras["graph"]) == {
        "graphiti-core": "graphiti-core[kuzu]==0.30.2",
        "kuzu": "kuzu==0.11.3",
    }
    assert extras["all"] == ["docugraph[crawl,graph,cloud,ollama]"]
    assert "ladybug" not in extras["all"][0]

    # The backend is opt-in; this PR does not bump the version.
    assert project["version"] == "0.3.0"
    assert project["requires-python"] == ">=3.11,<3.14"


def test_urls_point_at_the_changelog_and_security_policy():
    urls = _project()["urls"]
    assert urls["Changelog"].endswith("/blob/main/CHANGELOG.md")
    assert "Security" in urls
