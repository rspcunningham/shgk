"""Build the corpus: fetch new packages, then rebuild the curated stages.

Free, deterministic and idempotent -- safe to run at any time. Translation is
deliberately not here; see translate.py.

    python build.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from shgk import db  # noqa: E402
from shgk.curation import rebuild_duplicates, rebuild_exclusions  # noqa: E402
from shgk.http import HttpClient  # noqa: E402
from shgk.ingest import ingest  # noqa: E402

DATABASE = db.DEFAULT_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, help="index pages to crawl (default: all)")
    parser.add_argument(
        "--refresh", action="store_true",
        help="re-parse every package, even ones whose page is unchanged",
    )
    parser.add_argument("--workers", type=int, default=8,
                        help="packages fetched concurrently")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="minimum seconds between request starts")
    args = parser.parse_args()

    db.initialize(DATABASE)
    with db.connect(DATABASE) as connection:
        print("stage 1: packages")
        with HttpClient(delay=args.delay) as client:
            counts = ingest(
                connection, client,
                pages=args.pages, refresh=args.refresh,
                workers=args.workers, progress=print,
            )
        reported = False
        for label in ("new", "updated", "unchanged", "skipped", "empty",
                      "parse_error", "fetch_error", "questions"):
            if counts.get(label):
                print(f"  {label:<28}{counts[label]:>10,}")
                reported = True
        if not reported:
            print(f"  {'already up to date':<28}")

        print("stage 2: exclusions")
        for reason, count in sorted(
            rebuild_exclusions(connection).items(), key=lambda item: -item[1]
        ):
            print(f"  {reason:<28}{count:>10,}")

        print("stage 3: duplicates")
        stats = rebuild_duplicates(connection)
        print(f"  {'duplicate groups':<28}{stats['groups']:>10,}")
        print(f"  {'non-canonical rows':<28}{stats['duplicates']:>10,}")
        connection.commit()

        print("stages")
        for view in ("questions", "questions_clean", "questions_canonical",
                     "questions_translated"):
            count = connection.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
            print(f"  {view:<28}{count:>10,}")
        pending = connection.execute(
            """
            SELECT COUNT(*) FROM questions_canonical AS q
            WHERE NOT EXISTS (
                SELECT 1 FROM translations AS t
                WHERE t.question_id = q.id AND t.content_hash = q.content_hash
            )
            """
        ).fetchone()[0]
        print(f"  {'awaiting translation':<28}{pending:>10,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
