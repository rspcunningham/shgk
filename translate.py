"""Translate canonical questions that have no current translation.

    python translate.py 100
    python translate.py 100 --workers 16
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from shgk.progress import Progress
from shgk.translation import (
    AgentsTranslationClient,
    TranslationPipeline,
    install_pooled_openai_client,
)
from shgk.translation.pipeline import RunResult

# Questions are translated on one event loop, so this bounds in-flight requests
# rather than any pool of workers. Each in-flight question holds at most one
# connection, and the SDK's pool allows 1000, so only go wider than that.
DEFAULT_CONCURRENCY = 32
SDK_CONNECTION_LIMIT = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", type=int, help="how many questions to translate")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"questions in flight at once (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--refresh", action="store_true", help="also redo translations already current"
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be at least 1")
    if args.concurrency < 1:
        parser.error("concurrency must be at least 1")
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
    print(
        f"       per question: input {usage.input_tokens / result.completed:,.0f}, "
        f"output {usage.output_tokens / result.completed:,.0f}"
    )


def main() -> int:
    args = parse_args()
    load_dotenv(".env.local", override=False)
    load_dotenv(".env", override=False)
    if args.concurrency >= SDK_CONNECTION_LIMIT:
        install_pooled_openai_client(max_connections=args.concurrency * 2)

    with Progress("questions") as progress:
        result = asyncio.run(
            TranslationPipeline().run(
                AgentsTranslationClient(),
                limit=args.count,
                offset=args.offset,
                refresh=args.refresh,
                fail_fast=args.fail_fast,
                concurrency=args.concurrency,
                progress=progress,
            )
        )
    report(result)
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
