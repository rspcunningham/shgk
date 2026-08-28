"""Fetch new packages and rebuild the curated stages.

    python build.py
"""

from __future__ import annotations

import argparse

from shgk import corpus
from shgk.http import HttpClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, help="index pages to crawl (default: all)")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-parse every package, even ones whose page is unchanged",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="packages fetched concurrently"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, help="minimum seconds between requests"
    )
    return parser.parse_args()


def row(label: str, value: int | str) -> None:
    formatted = f"{value:,}" if isinstance(value, int) else value
    print(f"  {label:<28}{formatted:>10}")


FETCH_LABELS = (
    "new", "updated", "unchanged", "recovered", "skipped",
    "empty", "parse_error", "fetch_error", "questions",
)


def report(built: corpus.BuildReport) -> None:
    print("stage 1: packages")
    counted = [(name, built.fetched[name]) for name in FETCH_LABELS
               if built.fetched[name]]
    for label, value in counted or [("already up to date", "")]:
        row(label, value)

    print("stage 2: exclusions")
    for reason, count in sorted(built.exclusions.items(), key=lambda item: -item[1]):
        row(reason, count)

    print("stage 3: duplicates")
    row("duplicate groups", built.duplicate_groups)
    row("non-canonical rows", built.duplicate_rows)

    print("corpus")
    row("questions", built.stats.questions)
    row("clean", built.stats.clean)
    row("canonical", built.stats.canonical)
    row("translated", built.stats.translated)
    row("awaiting translation", built.stats.awaiting_translation)


def main() -> int:
    args = parse_args()
    with HttpClient(delay=args.delay) as client:
        built = corpus.build(
            client, pages=args.pages, refresh=args.refresh, workers=args.workers,
            progress=print,
        )
    report(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
