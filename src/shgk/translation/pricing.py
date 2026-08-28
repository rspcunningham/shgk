"""What a translation run cost.

Cost is derived rather than stored: the database records token counts, which do
not go stale, while prices do. Recording dollars at write time would freeze
whatever rate happened to apply that day.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import UsageTotals


@dataclass(frozen=True, slots=True)
class Rates:
    """USD per million tokens."""

    input: float
    cached_input: float
    cache_write: float
    output: float


# gpt-5.6-luna, short-context tier, as published 2026-08-27. Prompts here run to
# a few thousand tokens, far below the long-context threshold, so the short
# rates are the ones that apply.
LUNA = Rates(input=0.20, cached_input=0.02, cache_write=0.25, output=1.20)

PER_MILLION = 1_000_000


def cost(usage: UsageTotals, rates: Rates = LUNA) -> float:
    """The USD cost of the tokens in `usage`.

    Input divides into three separately priced buckets: tokens read from the
    prompt cache, tokens written to it, and the rest at the standard rate.
    Reasoning tokens are not added on top -- they are already inside
    output_tokens, and counting them again would roughly double the figure.
    """
    uncached = max(
        0,
        usage.input_tokens - usage.cached_input_tokens - usage.cache_write_input_tokens,
    )
    return (
        uncached * rates.input
        + usage.cached_input_tokens * rates.cached_input
        + usage.cache_write_input_tokens * rates.cache_write
        + usage.output_tokens * rates.output
    ) / PER_MILLION
