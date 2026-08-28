"""Translate canonical questions that have no current translation.

    python translate.py 100
    python translate.py 100 --workers 16
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from shgk import db
from shgk.translation import (
    AgentsTranslationClient,
    TranslationPipeline,
    install_pooled_openai_client,
)

# Rough planning figure only. Actual spend is dominated by reasoning tokens,
# which vary several-fold between questions; the run reports real usage.
COST_PER_QUESTION = 0.03


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", type=int, help="how many questions to translate")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--refresh", action="store_true", help="also redo translations already current"
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be at least 1")

    load_dotenv(".env.local", override=False)
    load_dotenv(".env", override=False)
    if args.workers > 8:
        install_pooled_openai_client()

    print(f"translating up to {args.count:,} questions "
          f"(rough estimate ${args.count * COST_PER_QUESTION:,.2f})")
    result = asyncio.run(
        TranslationPipeline(db.DEFAULT_PATH).run(
            AgentsTranslationClient(),
            limit=args.count,
            offset=args.offset,
            refresh=args.refresh,
            fail_fast=args.fail_fast,
            workers=args.workers,
            progress=lambda message: print(message, flush=True),
        )
    )
    print(
        f"selected={result.selected} completed={result.completed} "
        f"errors={result.errors}"
    )
    if result.completed:
        _report_usage(result.translated_ids)
    return 1 if result.errors else 0


def _report_usage(question_ids: list[int]) -> None:
    """Report what the run actually consumed, since estimates are unreliable here."""
    if not question_ids:
        return
    placeholders = ",".join("?" * len(question_ids))
    with db.connect(db.DEFAULT_PATH, read_only=True) as connection:
        row = connection.execute(
            f"""
            SELECT SUM(input_tokens) i, SUM(cached_input_tokens) c,
                   SUM(output_tokens) o, SUM(reasoning_output_tokens) r,
                   SUM(api_requests) q, COUNT(*) n
            FROM translations WHERE question_id IN ({placeholders})
            """,
            question_ids,
        ).fetchone()
    n = row["n"] or 1
    print(
        f"usage: {row['q']:,} requests, "
        f"input {row['i']:,} ({row['c']:,} cached), "
        f"output {row['o']:,} ({row['r']:,} reasoning)"
    )
    print(f"       per question: input {row['i'] / n:,.0f}, output {row['o'] / n:,.0f}")


if __name__ == "__main__":
    raise SystemExit(main())
