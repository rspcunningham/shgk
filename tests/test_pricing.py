from __future__ import annotations

from shgk.translation.models import UsageTotals
from shgk.translation.pricing import LUNA, Rates, cost

ROUND = Rates(input=1.0, cached_input=0.1, cache_write=2.0, output=10.0)


def _usage(**fields: int) -> UsageTotals:
    return UsageTotals(**fields)


def test_each_input_bucket_is_priced_separately() -> None:
    usage = _usage(
        input_tokens=1_000_000,
        cached_input_tokens=600_000,
        cache_write_input_tokens=300_000,
        output_tokens=0,
    )
    # 100k uncached at 1.0, 600k cached at 0.1, 300k written at 2.0
    assert cost(usage, ROUND) == 0.1 + 0.06 + 0.6


def test_reasoning_tokens_are_not_charged_twice() -> None:
    """They are already inside output_tokens; adding them would double the bill."""
    without = _usage(input_tokens=0, output_tokens=1_000_000)
    with_reasoning = _usage(
        input_tokens=0, output_tokens=1_000_000, reasoning_output_tokens=980_000
    )
    assert cost(without, ROUND) == cost(with_reasoning, ROUND) == 10.0


def test_a_fully_cached_prompt_costs_almost_nothing() -> None:
    cold = _usage(input_tokens=1_000_000, output_tokens=0)
    warm = _usage(
        input_tokens=1_000_000, cached_input_tokens=1_000_000, output_tokens=0
    )
    assert cost(warm, ROUND) < cost(cold, ROUND) / 5


def test_buckets_that_overrun_the_total_do_not_go_negative() -> None:
    """Guards against a provider reporting inconsistent counts."""
    usage = _usage(
        input_tokens=100,
        cached_input_tokens=90,
        cache_write_input_tokens=90,
        output_tokens=0,
    )
    assert cost(usage, ROUND) >= 0


def test_no_usage_costs_nothing() -> None:
    assert cost(UsageTotals(), LUNA) == 0.0


def test_luna_rates_are_the_published_ones() -> None:
    assert (LUNA.input, LUNA.cached_input, LUNA.cache_write, LUNA.output) == (
        0.20,
        0.02,
        0.25,
        1.20,
    )
