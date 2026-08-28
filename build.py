"""Fetch new packages and rebuild the curated stages.

    python build.py
"""

from __future__ import annotations

import argparse
import sys

from shgk import corpus
from shgk.http import HttpClient
from shgk.ingest import IndexUnavailable
from shgk.progress import Progress


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

    print("stage 3: canonical")
    row("merged from reprints", built.canonical["merged"])
    row("reprints folded in", built.canonical["reprints"])
    row("translations dropped", built.translations_dropped)

    print("corpus")
    row("questions", built.stats.questions)
    row("clean", built.stats.clean)
    row("canonical", built.stats.canonical)
    row("translated", built.stats.translated)
    row("awaiting translation", built.stats.awaiting_translation)


def main() -> int:
    args = parse_args()
    try:
        with HttpClient(delay=args.delay) as client, Progress("packages") as progress:
            built = corpus.build(
                client, pages=args.pages, refresh=args.refresh,
                workers=args.workers, progress=progress,
            )
    except IndexUnavailable as error:
        print(f"build: {error}", file=sys.stderr)
        return 1
    report(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
