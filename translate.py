"""Translate a random sample of canonical questions that have no translation.

    python translate.py 100
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from shgk.progress import Progress
from shgk.translation import AgentsTranslationClient, TranslationPipeline
from shgk.translation.pipeline import RunResult
from shgk.translation.pricing import cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", type=int, help="how many questions to translate")
    parser.add_argument(
        "--refresh", action="store_true", help="also redo translations already current"
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be at least 1")
    return args


def report(result: RunResult) -> None:
    print(
        f"selected={result.selected} completed={result.completed} "
        f"errors={result.errors}"
    )
    if not result.completed:
        return
    usage = result.usage
    print(
        f"usage: {usage.requests:,} requests, "
        f"input {usage.input_tokens:,} ({usage.cached_input_tokens:,} cached), "
        f"output {usage.output_tokens:,} ({usage.reasoning_output_tokens:,} reasoning)"
    )
    spent = cost(usage)
    print(
        f"       per question: input {usage.input_tokens / result.completed:,.0f}, "
        f"output {usage.output_tokens / result.completed:,.0f}, "
        f"${spent / result.completed:.4f}"
    )
    print(f"cost:  ${spent:,.2f}")


def main() -> int:
    args = parse_args()
    load_dotenv(".env.local", override=False)
    load_dotenv(".env", override=False)

    with Progress("questions") as progress:
        result = asyncio.run(
            TranslationPipeline().run(
                AgentsTranslationClient(),
                limit=args.count,
                refresh=args.refresh,
                fail_fast=args.fail_fast,
                    progress=progress,
            )
        )
    report(result)
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
