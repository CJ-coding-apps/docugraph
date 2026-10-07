#!/usr/bin/env python
"""One tiny call per cloud provider, against the real APIs.

Everything else in this repository runs without a network. The unit tests that
cover request shape use a mocked transport, which is what makes them fast and
deterministic -- and also what makes them unable to answer the one question that
only a live call can: does the model the operator configured still exist, and
does the provider still accept the request this package builds for it? A model
id is retired by the vendor on the vendor's schedule, and no amount of local
testing notices.

So this is deliberately *not* part of CI's test jobs. It needs secrets, it costs
a fraction of a cent, and it fails for reasons that have nothing to do with the
commit under test. It runs weekly and on demand, on `main` only, via
`.github/workflows/live-provider-smoke.yml`.

The model comes from an environment variable the workflow reads out of the
repository's *variables* (not secrets -- a model id is not a credential, and
being able to see it makes a failure readable). The key comes from a secret.
A provider with no key configured is reported as skipped, never as passed:
silence is not success, and a workflow that goes green because it tested
nothing is worse than one that never ran.

Run it locally with the same variables:

    OPENAI_API_KEY=sk-... OPENAI_MODEL=<a model id> \\
        .venv/bin/python scripts/live_provider_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Short enough that a provider billing by the token is not worth thinking about,
# and unambiguous enough that a wrong answer is visible at a glance.
PROMPT = "Reply with the single word: ok"
MAX_TOKENS = 16

ProviderCheck = tuple[str, str, str]  # (provider, verdict, detail)
VERDICT_OK = "ok"
VERDICT_SKIPPED = "skipped"
VERDICT_FAILED = "FAILED"


async def _check_llm(
    provider_name: str,
    key_var: str,
    model_var: str,
    build: object,
) -> ProviderCheck:
    """Construct one LLM provider and ask it for one word."""
    key = os.environ.get(key_var)
    model = os.environ.get(model_var)

    if not key:
        return provider_name, VERDICT_SKIPPED, f"{key_var} is not set"
    if not model:
        # A key with no model is the state this package now refuses to guess
        # its way out of, so the smoke check must not guess either.
        return (
            provider_name,
            VERDICT_SKIPPED,
            f"{model_var} is not set (set it in the repository's variables)",
        )

    try:
        provider = build(model=model, api_key=key)  # type: ignore[operator]
        text = await provider.generate(PROMPT, max_tokens=MAX_TOKENS)
    except Exception as e:  # noqa: BLE001 -- any failure here is the result
        return provider_name, VERDICT_FAILED, f"{type(e).__name__}: {e}"

    if not text.strip():
        return provider_name, VERDICT_FAILED, "the provider returned no text"
    return provider_name, VERDICT_OK, f"model={model!r} replied {text.strip()[:40]!r}"


def _check_cohere(model_var: str) -> ProviderCheck:
    """One embedding call. Cohere is reachable in this package for embeddings only."""
    key = os.environ.get("COHERE_API_KEY")
    model = os.environ.get(model_var)

    if not key:
        return "cohere (embeddings)", VERDICT_SKIPPED, "COHERE_API_KEY is not set"
    if not model:
        return (
            "cohere (embeddings)",
            VERDICT_SKIPPED,
            f"{model_var} is not set (set it in the repository's variables)",
        )

    try:
        from docugraph.core.embeddings import CohereEmbedder

        embedder = CohereEmbedder(model_name=model, api_key=key)
        vectors = embedder.embed(["ok"])
    except Exception as e:  # noqa: BLE001 -- any failure here is the result
        return "cohere (embeddings)", VERDICT_FAILED, f"{type(e).__name__}: {e}"

    if not vectors or not vectors[0]:
        return "cohere (embeddings)", VERDICT_FAILED, "the provider returned no vector"
    # The dimension is reported because it is measured, not tabulated: seeing a
    # number here is the live confirmation that the measurement path works.
    return "cohere (embeddings)", VERDICT_OK, f"model={model!r} returned {len(vectors[0])} dims"


async def main() -> int:
    from docugraph.core.llm import AnthropicLLM, OpenAILLM

    checks: list[ProviderCheck] = [
        await _check_llm("openai", "OPENAI_API_KEY", "OPENAI_MODEL", OpenAILLM),
        await _check_llm("anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", AnthropicLLM),
        _check_cohere("COHERE_MODEL"),
    ]

    width = max(len(name) for name, _, _ in checks)
    for provider_name, verdict, detail in checks:
        print(f"{provider_name:<{width}}  {verdict:<8}  {detail}")

    failed = [name for name, verdict, _ in checks if verdict == VERDICT_FAILED]
    skipped = [name for name, verdict, _ in checks if verdict == VERDICT_SKIPPED]
    print(
        f"\n{len(checks) - len(failed) - len(skipped)} passed, {len(skipped)} skipped, {len(failed)} failed"
    )
    if skipped:
        print(f"skipped (no key or no model configured): {', '.join(skipped)}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    if len(skipped) == len(checks):
        print("Nothing was checked. Configure at least one provider.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
