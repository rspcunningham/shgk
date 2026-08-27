"""Translate canonical questions that have no current translation.

This is the stage that costs money -- roughly $0.03 a question -- so the number
of questions is required rather than defaulted.

    python translate.py 100
    python translate.py 100 --workers 16
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv  # noqa: E402

from shgk import db  # noqa: E402
from shgk.translation import (  # noqa: E402
    AgentsTranslationClient,
    TranslationPipeline,
    install_pooled_openai_client,
)

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
          f"(~${args.count * COST_PER_QUESTION:,.0f})")
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
        f"selected={result['selected']} completed={result['completed']} "
        f"errors={result['errors']}"
    )
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
