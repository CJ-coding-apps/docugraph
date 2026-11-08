"""The one place that explains a missing optional dependency.

A core install does not include crawl4ai, Playwright, graphiti-core or Kuzu.
Features that need one of them import it lazily; when that import fails, this
module turns the ``ImportError`` into one sentence naming the extra that
supplies it, so a user never has to read a module name and guess.
"""

from __future__ import annotations

import importlib

__all__ = ["MissingExtraError", "require"]


class MissingExtraError(ImportError):
    """A feature was used without the extra that installs it.

    An ``ImportError`` subclass, so code that already treats "not installed" as
    non-fatal keeps doing so -- but the message says what to install instead of
    naming a module the user has never heard of.
    """


def _message(extra: str) -> str:
    message = f"This feature needs the '{extra}' extra: pip install 'docugraph[{extra}]'"
    if extra == "crawl":
        message += ", then: playwright install chromium"
    return message


def require(extra: str, module: str) -> None:
    """Import ``module``, or raise :class:`MissingExtraError` naming ``extra``.

    The import is the check. Probing with ``find_spec`` cannot see a missing
    *transitive* dependency (crawl4ai installed, Playwright absent), and it
    would raise only from the call site that remembered to probe, not from the
    one that needs the feature. Importing the module that carries the feature
    fails exactly when the feature cannot work.

    Args:
        extra: The install extra that supplies the module, e.g. ``"graph"``.
        module: An importable module name, e.g. ``"graphiti_core"``.
    """
    try:
        importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(_message(extra)) from exc
