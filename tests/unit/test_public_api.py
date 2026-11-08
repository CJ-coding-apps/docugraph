"""The declared Python surface.

ORACLE: WP-docugraph-0.3.0 §3.B4 — ``__all__`` is the supported surface, lazy
exports appear in ``dir()``, an unknown name raises the standard
``AttributeError``, and a resolved lazy name is cached.
"""

from __future__ import annotations

import pytest

EXPECTED_ALL = {
    "__version__",
    "Config",
    "get_config",
    "Document",
    "Chunk",
    "Entity",
    "Relationship",
    "MemoryEntry",
    "DocCrawler",
    "Chunker",
    "VectorStore",
    "HybridRetriever",
    "SearchMode",
    "FusionStrategy",
    "UrlPolicy",
    "MissingExtraError",
}

LAZY_NAMES = (
    "DocCrawler",
    "Chunker",
    "VectorStore",
    "HybridRetriever",
    "SearchMode",
    "FusionStrategy",
)


def test_all_is_the_supported_surface():
    import docugraph

    assert set(docugraph.__all__) == EXPECTED_ALL


def test_every_declared_name_resolves():
    import docugraph

    for name in docugraph.__all__:
        assert getattr(docugraph, name) is not None


def test_dir_lists_every_exported_name():
    import docugraph

    listing = dir(docugraph)
    assert set(docugraph.__all__) <= set(listing)


def test_unknown_name_raises_the_standard_attribute_error():
    import docugraph

    with pytest.raises(AttributeError) as info:
        _ = docugraph.no_such_public_name
    assert "module 'docugraph' has no attribute 'no_such_public_name'" in str(info.value)


def test_lazy_names_resolve_and_are_cached():
    import docugraph

    for name in LAZY_NAMES:
        value = getattr(docugraph, name)
        assert value is not None
        # A resolved lazy export is cached in the package namespace, so the
        # second access cannot disagree with the first.
        assert vars(docugraph).get(name) is value


def test_public_models_are_the_expected_types():
    from pydantic import BaseModel

    import docugraph

    for name in ("Document", "Chunk", "Entity", "Relationship", "MemoryEntry"):
        assert issubclass(getattr(docugraph, name), BaseModel)
